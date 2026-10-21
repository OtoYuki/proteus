//! `proteus film` — a short captioned film (MP4) and a still (PNG) of what each candidate
//! changed against its parent, with the diff as JSON next to them. Frames come from
//! `proteus_render::film`; this module renders them in parallel and streams them, in order, to
//! `ffmpeg`.

use super::analyze::collect_inputs;
use super::diff::prepare;
use super::prelude::*;
use clap::ValueEnum;
use proteus_core::variant::{diff, DiffOptions};
use proteus_render::film::{CpuWorker, Film, FilmSettings};
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Condvar, Mutex};

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
pub enum Aspect {
    /// 1080 × 1080: LinkedIn, Bluesky and Mastodon feeds
    #[value(name = "1:1")]
    Square,
    /// 1080 × 1350: the tallest feed format
    #[value(name = "4:5")]
    Portrait,
    /// 1920 × 1080: slides and talks
    #[value(name = "16:9")]
    Wide,
}

impl Aspect {
    fn size(self) -> (usize, usize) {
        match self {
            Aspect::Square => (1080, 1080),
            Aspect::Portrait => (1080, 1350),
            Aspect::Wide => (1920, 1080),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
pub enum Renderer {
    /// The GPU when a hardware adapter is present, else the CPU
    Auto,
    /// The GPU, or fail
    Gpu,
    /// The CPU rasteriser: the reference, the same bytes on every machine
    Cpu,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
pub enum Encoder {
    /// NVIDIA's hardware encoder when it works on this machine, else libx264
    Auto,
    Nvenc,
    X264,
}

/// Arguments of `proteus film`.
#[derive(clap::Args, Debug)]
#[command(after_help = "Examples:
  proteus film --parent wt.pdb design_07.pdb
  proteus film --parent 1b27.pdb 1b3s.pdb --chains A,D --aspect 4:5 --out films/
  proteus film --parent scaffold.cif designs/ --still-only

Needs ffmpeg on PATH for the MP4; without it the still and the JSON are still written.")]
pub struct Args {
    /// The structure the candidates are compared with
    #[arg(long, value_name = "FILE")]
    pub parent: PathBuf,

    /// Candidate structures or directories of them
    #[arg(value_name = "CANDIDATE", required = true)]
    pub candidates: Vec<PathBuf>,

    /// Compare and film only these chains, in both structures
    #[arg(long, value_delimiter = ',', value_name = "IDS")]
    pub chains: Option<Vec<String>>,

    /// Directory for `<candidate>.mp4`, `.png` and `.json`
    #[arg(short, long, default_value = "films", value_name = "DIR")]
    pub out: PathBuf,

    #[arg(long, value_enum, default_value_t = Aspect::Square)]
    pub aspect: Aspect,

    #[arg(long, default_value_t = 30)]
    pub fps: u32,

    /// Multiplies every shot's length (2 = half speed)
    #[arg(long, default_value_t = 1.0)]
    pub pace: f32,

    /// Site groups to visit, most changed first
    #[arg(long, default_value_t = 3)]
    pub max_sites: usize,

    /// Samples per pixel along each axis (2 = 4 samples)
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u8).range(1..=4))]
    pub supersample: u8,

    /// Take the still at this many seconds into the film [default: the middle of the first
    /// site]
    #[arg(long, value_name = "SECONDS")]
    pub still_at: Option<f32>,

    /// Write the still and the JSON only, no MP4
    #[arg(long)]
    pub still_only: bool,

    #[arg(long, value_enum, default_value_t = Encoder::Auto)]
    pub encoder: Encoder,

    #[arg(long, value_enum, default_value_t = Renderer::Auto)]
    pub renderer: Renderer,

    /// Frames rendered in parallel [default: the number of CPUs]
    #[arg(short, long)]
    pub jobs: Option<usize>,

    /// See `proteus diff --local-radius`
    #[arg(long, default_value_t = proteus_core::variant::DEFAULT_LOCAL_RADIUS, value_name = "Å")]
    pub local_radius: f64,
}

pub async fn run(args: Args) -> Result<()> {
    if !(args.pace > 0.05 && args.pace <= 20.0) {
        bail!("--pace must be between 0.05 and 20");
    }
    if !(1..=120).contains(&args.fps) {
        bail!("--fps must be between 1 and 120");
    }
    let opts = DiffOptions {
        chains: args.chains.clone(),
        local_radius: args.local_radius,
    };
    let parent = prepare(&args.parent, &opts)?;
    let (files, unreadable) = collect_inputs(&args.candidates)?;
    for (p, why) in &unreadable {
        eprintln!("skipped {}: {why}", p.display());
    }
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("creating {}", args.out.display()))?;
    let encoder = if args.still_only {
        None
    } else {
        match pick_encoder(args.encoder) {
            Ok(e) => Some(e),
            // Asked for by name: say so and stop rather than quietly writing no MP4.
            Err(e) if args.encoder != Encoder::Auto => {
                return Err(e.context("pass --still-only to skip the MP4"));
            }
            Err(e) => {
                eprintln!("no MP4: {e:#}; writing the stills and the JSON only");
                None
            }
        }
    };
    let threads = args
        .jobs
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()))
        .max(1);
    let parent_stem = stem(&args.parent);
    let names = output_names(&files);
    let mut failed = unreadable.len();
    for (f, name) in files.iter().zip(&names) {
        match film_one(
            f,
            name,
            &parent,
            &parent_stem,
            &opts,
            &args,
            encoder.as_deref(),
            threads,
        ) {
            Ok(written) => println!("{}", written.join("  ")),
            Err(e) => {
                failed += 1;
                eprintln!("{}: {e:#}", f.display());
            }
        }
    }
    if failed > 0 {
        bail!("{failed} input(s) could not be filmed");
    }
    Ok(())
}

fn stem(p: &Path) -> String {
    let name = p
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let name = name.strip_suffix(".gz").unwrap_or(&name).to_string();
    match name.rsplit_once('.') {
        Some((s, _)) => s.to_string(),
        None => name,
    }
}

/// The output file name (without extension) for each input: its stem, made unique. Chai-1 names
/// its samples `pred.model_idx_0.cif` … and a local AlphaFold 3 run puts every sample in its own
/// directory as `model.cif`, so neither the bare stem nor `Path::with_extension` (which would cut
/// `pred.model_idx_0` back to `pred`) is enough. A clashing stem takes its directory's name in
/// front; one that still clashes takes its position.
fn output_names(files: &[PathBuf]) -> Vec<String> {
    use std::collections::HashMap;
    let count = |names: &[String]| {
        let mut n: HashMap<&str, usize> = HashMap::new();
        for s in names {
            *n.entry(s.as_str()).or_default() += 1;
        }
        names.iter().map(|s| n[s.as_str()]).collect::<Vec<_>>()
    };
    let mut names: Vec<String> = files.iter().map(|f| stem(f)).collect();
    let dup = count(&names);
    for (i, f) in files.iter().enumerate() {
        if dup[i] > 1 {
            if let Some(d) = f.parent().and_then(|d| d.file_name()) {
                names[i] = format!("{}_{}", d.to_string_lossy(), names[i]);
            }
        }
    }
    // A position can still meet a name already taken (`r_m_1.pdb` beside two `r/m.pdb`), so a
    // clashing name takes the first free number from its position on.
    let dup = count(&names);
    let mut used: std::collections::HashSet<String> = names
        .iter()
        .zip(&dup)
        .filter(|(_, d)| **d == 1)
        .map(|(n, _)| n.clone())
        .collect();
    for (i, n) in names.iter_mut().enumerate() {
        if dup[i] > 1 {
            let mut k = i + 1;
            while !used.insert(format!("{n}_{k}")) {
                k += 1;
            }
            *n = format!("{n}_{k}");
        }
    }
    names
}

/// Fewest paired residues, as a fraction of the larger structure, that `film` accepts.
const MIN_COVERAGE: f64 = 0.5;

#[allow(clippy::too_many_arguments)]
fn film_one(
    path: &Path,
    out_name: &str,
    parent: &proteus_core::variant::PreparedStructure,
    parent_stem: &str,
    opts: &DiffOptions,
    args: &Args,
    encoder: Option<&[String]>,
    threads: usize,
) -> Result<Vec<String>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let name = path.file_name().and_then(|n| n.to_str());
    let candidate =
        proteus_core::variant::PreparedStructure::from_bytes(&bytes, name, opts.chains.as_deref())?;
    let d = diff(parent, &candidate, opts)?;
    // A film of "what changed" against the wrong parent would caption hundreds of sites that
    // are not mutations. `diff` can still show the pairing.
    if d.coverage < MIN_COVERAGE {
        bail!(
            "only {:.0} % of residues pair with the parent ({}, identity {:.0} %): not the same \
             protein, or the wrong chains; `proteus diff` shows the pairing",
            d.coverage * 100.0,
            d.pairing,
            d.identity * 100.0
        );
    }
    let cstem = stem(path);
    let (width, height) = args.aspect.size();
    let settings = FilmSettings {
        width,
        height,
        fps: args.fps,
        pace: args.pace,
        max_groups: args.max_sites,
        supersample: args.supersample as usize,
        title: format!("{cstem}  vs  {parent_stem}"),
    };
    let film = Film::new(&bytes, name, opts.chains.as_deref(), &d, settings)?;
    let out_path = |ext: &str| args.out.join(format!("{out_name}.{ext}"));
    let mut written = Vec::new();

    let json_path = out_path("json");
    let mut v = serde_json::to_value(&d)?;
    v["candidate"] = serde_json::Value::String(path.display().to_string());
    v["parent"] = serde_json::Value::String(args.parent.display().to_string());
    std::fs::write(&json_path, serde_json::to_string_pretty(&v)?)?;
    written.push(json_path.display().to_string());

    let png_path = out_path("png");
    let mut rgb = vec![0u8; width * height * 3];
    let still = match args.still_at {
        Some(t) => film.frame_at(t),
        None => film.still_frame(),
    };
    film.render_cpu(still, &mut CpuWorker::new(), &mut rgb);
    write_png(&png_path, width, height, &rgb)?;
    written.push(png_path.display().to_string());

    if let Some(enc) = encoder {
        let mp4_path = out_path("mp4");
        encode(&film, enc, &mp4_path, threads, args.renderer)?;
        written.push(mp4_path.display().to_string());
    }
    Ok(written)
}

fn write_png(path: &Path, width: usize, height: usize, rgb: &[u8]) -> Result<()> {
    let file =
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), width as u32, height as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    // sRGB, which is what the frame's colours are.
    enc.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    let mut w = enc.write_header()?;
    w.write_image_data(rgb)?;
    Ok(())
}

/// ffmpeg output arguments for each encoder: H.264 (every feed and player takes it), 4:2:0,
/// tagged BT.709, `faststart` so a feed can begin playing before the file has downloaded.
fn encoder_args(e: Encoder) -> Vec<String> {
    let codec: &[&str] = match e {
        Encoder::Nvenc => &[
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p7",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            "18",
            "-b:v",
            "0",
            "-profile:v",
            "high",
        ],
        Encoder::X264 | Encoder::Auto => &["-c:v", "libx264", "-preset", "slow", "-crf", "17"],
    };
    let mut v: Vec<String> = codec.iter().map(|s| s.to_string()).collect();
    v.extend(
        [
            "-vf",
            "scale=out_color_matrix=bt709:out_range=tv",
            "-pix_fmt",
            "yuv420p",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            // NVENC ignores -color_primaries/-color_trc; writing the VUI in the bitstream works
            // for both encoders, so players do not have to guess.
            "-bsf:v",
            "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1",
            "-movflags",
            "+faststart",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    v
}

/// The encoder to use: NVENC when asked or when `auto` and a one-frame test encode succeeds (a
/// listed encoder can still fail: no NVIDIA GPU, a driver mismatch, every session busy), else
/// libx264.
fn pick_encoder(choice: Encoder) -> Result<Vec<String>> {
    let probe = |e: Encoder| -> bool {
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i"])
            .arg("color=c=black:s=256x256:d=0.1")
            .args(encoder_args(e))
            .args(["-f", "null", "-"])
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        cmd.status().map(|s| s.success()).unwrap_or(false)
    };
    if Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_err()
    {
        bail!("ffmpeg is not on PATH");
    }
    let chosen = match choice {
        Encoder::Nvenc if probe(Encoder::Nvenc) => Encoder::Nvenc,
        Encoder::Nvenc => bail!("h264_nvenc does not work here (test encode failed)"),
        Encoder::Auto if probe(Encoder::Nvenc) => Encoder::Nvenc,
        Encoder::X264 | Encoder::Auto if probe(Encoder::X264) => Encoder::X264,
        _ => bail!("neither h264_nvenc nor libx264 works in this ffmpeg"),
    };
    eprintln!(
        "encoding with {}",
        if chosen == Encoder::Nvenc {
            "h264_nvenc (GPU)"
        } else {
            "libx264"
        }
    );
    Ok(encoder_args(chosen))
}

/// Render every frame of `film` on `threads` threads and pipe them, in order, to ffmpeg.
fn encode(
    film: &Film,
    enc: &[String],
    out: &Path,
    threads: usize,
    renderer: Renderer,
) -> Result<()> {
    let s = film.settings();
    let (w, h) = (s.width, s.height);
    let mut child = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo"])
        .args(["-pix_fmt", "rgb24", "-s", &format!("{w}x{h}")])
        .args(["-r", &s.fps.to_string(), "-i", "-"])
        .args(enc)
        .arg(out)
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("starting ffmpeg")?;
    let mut stdin = child.stdin.take().context("ffmpeg stdin")?;
    let frames = film.frame_count();
    let bar = ProgressBar::new(frames as u64);
    bar.set_style(
        ProgressStyle::with_template(&format!(
            "{{msg}} {{bar:30.{}}} {{pos}}/{{len}} frames {{elapsed}}",
            tint(proteus_render::brand::Role::Accent)
        ))
        .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    bar.set_message(
        out.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default(),
    );

    let mut sink = |frame: &[u8]| -> Result<()> {
        stdin.write_all(frame)?;
        bar.inc(1);
        Ok(())
    };
    let result = match gpu_renderer(film, renderer)? {
        Some(gpu) => render_gpu(film, gpu, &mut sink),
        None => render_in_order(film, threads, &mut sink),
    };
    drop(stdin);
    let output = child.wait_with_output().context("waiting for ffmpeg")?;
    bar.finish_and_clear();
    // ffmpeg exiting early makes the frame writes fail with a broken pipe; its own message says
    // why, so it goes first.
    if !output.status.success() {
        bail!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    result
}

#[cfg(feature = "gpu")]
type Gpu = proteus_render::film::gpu::GpuRenderer;
#[cfg(not(feature = "gpu"))]
type Gpu = std::convert::Infallible;

/// The GPU renderer the choice asks for, `None` for the CPU path.
fn gpu_renderer(film: &Film, choice: Renderer) -> Result<Option<Gpu>> {
    if choice == Renderer::Cpu {
        return Ok(None);
    }
    #[cfg(feature = "gpu")]
    {
        match Gpu::new(film) {
            Ok(g) => {
                eprintln!("rendering on {}", g.adapter());
                Ok(Some(g))
            }
            Err(why) if choice == Renderer::Gpu => bail!("--renderer gpu: {why}"),
            Err(why) => {
                eprintln!("rendering on the CPU: {why}");
                Ok(None)
            }
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = film;
        if choice == Renderer::Gpu {
            bail!("--renderer gpu: this build has no GPU support (cargo feature `gpu`)");
        }
        Ok(None)
    }
}

/// Frames from the GPU, one after another (the GPU is the parallel part), to `sink`.
#[cfg(feature = "gpu")]
fn render_gpu(film: &Film, mut gpu: Gpu, sink: &mut impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
    let s = film.settings();
    let mut rgb = vec![0u8; s.width * s.height * 3];
    let text = proteus_render::film::text::TextPainter::new();
    for i in 0..film.frame_count() {
        film.render_gpu(i, &mut gpu, &text, &mut rgb)
            .map_err(|e| anyhow::anyhow!(e))?;
        sink(&rgb)?;
    }
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn render_gpu(_: &Film, gpu: Gpu, _: &mut impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
    match gpu {}
}

/// Render frames on `threads` threads and hand them to `sink` in frame order. At most
/// `2 × threads` frames are in memory at once.
fn render_in_order(
    film: &Film,
    threads: usize,
    sink: &mut impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    let frames = film.frame_count();
    let s = film.settings();
    let frame_bytes = s.width * s.height * 3;
    let window = threads * 2;
    let next = AtomicUsize::new(0);
    let abort = AtomicBool::new(false);
    let delivered = (Mutex::new(0usize), Condvar::new());
    let (tx, rx) = mpsc::sync_channel::<(usize, Vec<u8>)>(window);
    std::thread::scope(|scope| -> Result<()> {
        for _ in 0..threads.min(frames.max(1)) {
            let tx = tx.clone();
            let (next, abort, delivered) = (&next, &abort, &delivered);
            scope.spawn(move || {
                let mut worker = CpuWorker::new();
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= frames || abort.load(Ordering::SeqCst) {
                        break;
                    }
                    {
                        let (lock, cv) = delivered;
                        let mut done = lock.lock().unwrap();
                        while i >= *done + window && !abort.load(Ordering::SeqCst) {
                            done = cv.wait(done).unwrap();
                        }
                    }
                    let mut rgb = vec![0u8; frame_bytes];
                    film.render_cpu(i, &mut worker, &mut rgb);
                    if tx.send((i, rgb)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        let mut pending = std::collections::BTreeMap::new();
        let mut want = 0usize;
        let mut err = None;
        for (i, rgb) in rx {
            pending.insert(i, rgb);
            while let Some(rgb) = pending.remove(&want) {
                if err.is_none() {
                    if let Err(e) = sink(&rgb) {
                        err = Some(e);
                        abort.store(true, Ordering::SeqCst);
                    }
                }
                want += 1;
                let (lock, cv) = &delivered;
                *lock.lock().unwrap() = want;
                cv.notify_all();
            }
        }
        match err {
            Some(e) => Err(e),
            None if want < frames => bail!("rendered {want} of {frames} frames"),
            None => Ok(()),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stems_drop_compression_and_format_extensions() {
        assert_eq!(stem(Path::new("a/1b3s.pdb")), "1b3s");
        assert_eq!(stem(Path::new("design_07.cif.gz")), "design_07");
        assert_eq!(stem(Path::new("noext")), "noext");
    }

    #[test]
    fn output_names_never_collide() {
        let names = |v: &[&str]| output_names(&v.iter().map(PathBuf::from).collect::<Vec<_>>());
        // Chai-1: dotted stems are kept whole (with_extension would make all five `pred`).
        assert_eq!(
            names(&["c/pred.model_idx_0.cif", "c/pred.model_idx_1.cif"]),
            ["pred.model_idx_0", "pred.model_idx_1"]
        );
        // Local AlphaFold 3: one `model.cif` per sample directory.
        assert_eq!(
            names(&[
                "j/seed-1_sample-0/model.cif",
                "j/seed-1_sample-1/model.cif",
                "x.pdb"
            ]),
            ["seed-1_sample-0_model", "seed-1_sample-1_model", "x"]
        );
        // Same directory name and stem: position decides.
        assert_eq!(
            names(&["a/run/m.pdb", "b/run/m.cif"]),
            ["run_m_1", "run_m_2"]
        );
        // A numbered name that is already some file's own stem is skipped.
        assert_eq!(
            names(&["r/m.pdb", "s/r/m.pdb", "r_m_1.pdb"]),
            ["r_m_2", "r_m_3", "r_m_1"]
        );
    }
}
