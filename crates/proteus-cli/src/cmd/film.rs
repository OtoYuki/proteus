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

    /// Directory for <candidate>.mp4, .png and .json
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
    let mut failed = unreadable.len();
    for f in &files {
        match film_one(
            f,
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

#[allow(clippy::too_many_arguments)]
fn film_one(
    path: &Path,
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
    let base = args.out.join(&cstem);
    let mut written = Vec::new();

    let json_path = base.with_extension("json");
    let mut v = serde_json::to_value(&d)?;
    v["candidate"] = serde_json::Value::String(path.display().to_string());
    v["parent"] = serde_json::Value::String(args.parent.display().to_string());
    std::fs::write(&json_path, serde_json::to_string_pretty(&v)?)?;
    written.push(json_path.display().to_string());

    let png_path = base.with_extension("png");
    let mut rgb = vec![0u8; width * height * 3];
    let still = match args.still_at {
        Some(t) => film.frame_at(t),
        None => film.still_frame(),
    };
    film.render_cpu(still, &mut CpuWorker::new(), &mut rgb);
    write_png(&png_path, width, height, &rgb)?;
    written.push(png_path.display().to_string());

    if let Some(enc) = encoder {
        let mp4_path = base.with_extension("mp4");
        encode(&film, enc, &mp4_path, threads)?;
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
fn encode(film: &Film, enc: &[String], out: &Path, threads: usize) -> Result<()> {
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

    let result = render_in_order(film, threads, |frame| {
        stdin.write_all(frame)?;
        bar.inc(1);
        Ok(())
    });
    drop(stdin);
    let output = child.wait_with_output().context("waiting for ffmpeg")?;
    bar.finish_and_clear();
    result?;
    if !output.status.success() {
        bail!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

/// Render frames on `threads` threads and hand them to `sink` in frame order. At most
/// `2 × threads` frames are in memory at once.
fn render_in_order(
    film: &Film,
    threads: usize,
    mut sink: impl FnMut(&[u8]) -> Result<()>,
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
}
