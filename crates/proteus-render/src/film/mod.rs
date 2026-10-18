//! `proteus film`: a short, captioned film of what a design candidate changed against its
//! parent, from a [`VariantDiff`].
//!
//! Every frame is a pure function of (structure, diff, settings, frame index): no clock, no
//! random numbers. The CPU path is the reference and its frames are the same bytes on every run.
//!
//! Shot list (durations × `pace`):
//! 1. establish: the whole candidate, a slow yaw, the changed sites lit;
//! 2. per site group (at most `max_groups`): fly in to the group, hold with a slow orbit while
//!    the side chains, the parent's removed atoms (ghosts) and the lost (dashed) and gained
//!    (solid) interactions grow in, a clipping slab opening the protein in front of the site;
//! 3. summary: pull back to the whole candidate with the diff's counts.
//!
//! Only *clear, local* changes are drawn or captioned (see `proteus_core::variant`): marginal
//! and distal changes are counted on the summary card and never shown as findings.

#[cfg(feature = "gpu")]
pub mod gpu;
pub mod text;

use crate::brand;
use crate::error::RenderError;
use crate::geometry::mesh::{
    generate_cartoon_mesh_detailed, generate_cylinder_mesh, generate_sphere_mesh, TriangleMesh,
};
use crate::rasterizer::buffer::{ColorRGB, Framebuffer};
use crate::rasterizer::camera::OrbitCamera;
use crate::rasterizer::pipeline::Rasterizer;
use crate::rasterizer::shader::{plddt_to_color, secondary_structure_to_color, ColorScheme};
use nalgebra::Vector3;
use proteus_core::structure::{assign_secondary_structure, SecondaryStructure};
use proteus_core::variant::{Change, InteractionKind, VariantDiff};
use std::collections::HashMap;
use text::{Canvas, TextPainter, Weight};

const fn hex(v: u32) -> ColorRGB {
    ColorRGB::new((v >> 16) as u8, (v >> 8) as u8, v as u8)
}

/// Film colours on the dark theme's ground. Lost and gained differ in pattern (dashed and
/// solid) as well as hue, so the distinction survives colour-vision deficiency and a mute
/// thumbnail.
mod palette {
    use super::hex;
    use crate::rasterizer::buffer::ColorRGB;
    pub const SITE: ColorRGB = hex(0xD9CE3F);
    pub const LOST: ColorRGB = hex(0xFF8F5E);
    pub const GAINED: ColorRGB = hex(0x8FE6D2);
    pub const GHOST: ColorRGB = hex(0x8A8D72);
    pub const CARBON: ColorRGB = hex(0xE3E1B6);
    pub const NITROGEN: ColorRGB = hex(0x7DB6E8);
    pub const OXYGEN: ColorRGB = hex(0xE0564A);
    pub const SULFUR: ColorRGB = hex(0xD9CE3F);
}

/// What the film looks like and how long it runs.
#[derive(Debug, Clone)]
pub struct FilmSettings {
    pub width: usize,
    pub height: usize,
    pub fps: u32,
    /// Multiplies every shot's duration.
    pub pace: f32,
    /// Site groups filmed, most changed first.
    pub max_groups: usize,
    /// Supersampling factor per axis (2 = 4 samples per pixel).
    pub supersample: usize,
    /// `1b3s vs 1b27`: the top-left line and the summary title.
    pub title: String,
}

impl Default for FilmSettings {
    fn default() -> Self {
        Self {
            width: 1080,
            height: 1080,
            fps: 30,
            pace: 1.0,
            max_groups: 3,
            supersample: 2,
            title: String::new(),
        }
    }
}

/// Seconds per shot at pace 1.
const ESTABLISH_S: f32 = 2.6;
const FLY_S: f32 = 2.0;
const HOLD_S: f32 = 3.4;
const SUMMARY_S: f32 = 2.6;
/// Yaw swept during an establish or a hold (radians, each side).
const SWEEP: f32 = 0.30;
/// A slight downward look reads as "from above", the usual way structures are shown.
const PITCH: f32 = 0.18;

#[derive(Debug, Clone)]
struct Atom {
    name: String,
    pos: Vector3<f32>,
}

fn element_color(atom: &str) -> ColorRGB {
    match atom.chars().next() {
        Some('N') => palette::NITROGEN,
        Some('O') => palette::OXYGEN,
        Some('S') => palette::SULFUR,
        _ => palette::CARBON,
    }
}

/// Covalent bonds within one residue's heavy atoms, by distance.
fn bonds_of(atoms: &[Atom]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for i in 0..atoms.len() {
        for j in (i + 1)..atoms.len() {
            let d = (atoms[i].pos - atoms[j].pos).norm();
            if d > 0.9 && d < 1.95 {
                out.push((i, j));
            }
        }
    }
    out
}

/// Side-chain atoms drawn as sticks: everything but the backbone N, C, O (proline keeps N, its
/// ring closes on it), plus any backbone atom an interaction uses, with its bonded neighbour.
fn shown_atoms(name: &str, all: &[Atom], extra: &[String]) -> Vec<Atom> {
    let mut keep: Vec<String> = all
        .iter()
        .map(|a| a.name.clone())
        .filter(|n| !matches!(n.as_str(), "N" | "C" | "O" | "OXT") || (name == "PRO" && n == "N"))
        .collect();
    for e in extra {
        match e.as_str() {
            "O" | "OXT" => keep.extend(["O".into(), "C".into(), "CA".into()]),
            "N" => keep.extend(["N".into(), "CA".into()]),
            "C" => keep.extend(["C".into(), "CA".into()]),
            other => keep.push(other.to_string()),
        }
    }
    all.iter()
        .filter(|a| keep.contains(&a.name))
        .cloned()
        .collect()
}

#[derive(Debug, Clone)]
struct Sticks {
    atoms: Vec<Atom>,
    bonds: Vec<(usize, usize)>,
    ghost: bool,
}

#[derive(Debug, Clone)]
struct Line {
    a: Vector3<f32>,
    b: Vector3<f32>,
    lost: bool,
}

#[derive(Debug, Clone, Default)]
struct Caption {
    title: String,
    /// Segments of the second line with their colours (lost and gained phrases differ).
    body: Vec<(String, ColorRGB)>,
    detail: String,
}

#[derive(Debug, Clone)]
struct Group {
    /// Ribbon residue indices of the group's sites.
    site_residues: Vec<usize>,
    /// C-alphas of the sites, marked in the establish and summary shots.
    site_cas: Vec<Vector3<f32>>,
    sticks: Vec<Sticks>,
    lines: Vec<Line>,
    view: View,
    caption: Caption,
}

#[derive(Debug, Clone, Copy)]
struct View {
    center: Vector3<f32>,
    yaw: f32,
    zoom: f32,
    /// 0 = the whole structure in its colours, 1 = context dimmed around the group.
    focus: f32,
    /// Å in front of `center` beyond which nothing is drawn.
    clip: f32,
    /// Vertical pan (Å) that keeps the subject above the caption panel.
    lift: f32,
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Geometric interpolation for scales and distances, so a zoom feels even.
fn glerp(a: f32, b: f32, t: f32) -> f32 {
    (a.max(1e-6).ln() + (b.max(1e-6).ln() - a.max(1e-6).ln()) * t).exp()
}

impl View {
    fn mix(a: &View, b: &View, t: f32) -> View {
        // The shorter way round.
        let mut dy = b.yaw - a.yaw;
        while dy > std::f32::consts::PI {
            dy -= std::f32::consts::TAU;
        }
        while dy < -std::f32::consts::PI {
            dy += std::f32::consts::TAU;
        }
        View {
            center: a.center + (b.center - a.center) * t,
            yaw: a.yaw + dy * t,
            zoom: glerp(a.zoom, b.zoom, t),
            focus: lerp(a.focus, b.focus, t),
            clip: glerp(a.clip, b.clip, t),
            lift: lerp(a.lift, b.lift, t),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Segment {
    Establish,
    Fly {
        from: Option<usize>,
        to: Option<usize>,
    },
    Hold(usize),
    Summary,
}

/// Everything a renderer needs for one frame's structure; captions are drawn afterwards by
/// [`Film::draw_overlay`], the same on every backend.
pub struct FrameScene {
    pub camera: OrbitCamera,
    /// Ribbon colour per residue.
    pub residue_colors: Vec<ColorRGB>,
    /// Sticks and interaction lines, by colour.
    pub overlays: Vec<(TriangleMesh, ColorRGB)>,
    pub clip_near: Option<f32>,
    pub depth_range: (f32, f32),
}

/// A film of one candidate, ready to render frame by frame. Immutable once built, so frames
/// can be rendered on many threads at once.
pub struct Film {
    settings: FilmSettings,
    ribbon: TriangleMesh,
    base_colors: Vec<ColorRGB>,
    camera: OrbitCamera,
    bounding_radius: f32,
    overview: View,
    groups: Vec<Group>,
    timeline: Vec<(Segment, usize, usize)>,
    frames: usize,
    establish: Caption,
    summary: Caption,
    /// Shown above the summary card: what is not on screen.
    footnote: String,
}

/// Ribbon residues of a filtered structure, keyed by chain and number.
fn residue_index(labels: &[(String, isize)]) -> HashMap<(String, isize), usize> {
    let mut map = HashMap::with_capacity(labels.len());
    for (i, k) in labels.iter().enumerate() {
        map.entry(k.clone()).or_insert(i);
    }
    map
}

fn v32(p: [f64; 3]) -> Vector3<f32> {
    Vector3::new(p[0] as f32, p[1] as f32, p[2] as f32)
}

fn counts_phrase(counts: &[(InteractionKind, usize)], sign: char) -> String {
    counts
        .iter()
        .filter(|(_, n)| *n > 0)
        .map(|(k, n)| format!("{sign}{n} {}", k.label(*n)))
        .collect::<Vec<_>>()
        .join("  ")
}

impl Film {
    /// Build the film of `candidate` (PDB/mmCIF bytes; `hint` is its file name) against the
    /// parent described by `diff`, which must have been computed on the same file and chains.
    pub fn new(
        candidate: &[u8],
        hint: Option<&str>,
        chains: Option<&[String]>,
        diff: &VariantDiff,
        settings: FilmSettings,
    ) -> Result<Film, RenderError> {
        let mut pdb = proteus_core::io::open_structure_bytes(candidate, hint)
            .map_err(|e| RenderError::PdbParse(e.to_string()))?;
        if let Some(keep) = chains {
            pdb.remove_chains_by(|c| !keep.iter().any(|k| k == c.id().trim()));
        }
        let trace = crate::trace_of(&pdb);
        if trace.ca.len() < 2 {
            return Err(RenderError::PdbParse(
                "the candidate has fewer than 2 C-alpha residues to film".into(),
            ));
        }
        let ss =
            assign_secondary_structure(&proteus_core::backbone::extract_backbone(&trace.protein));

        // A finer ribbon than the terminal's: 10 spline points per residue, 16-sided profile.
        let mut ribbon = TriangleMesh::new();
        let n = trace.ca.len();
        let mut start = 0;
        for end in 1..=n {
            if end == n || trace.breaks[end] {
                if end - start >= 2 {
                    let mut part = generate_cartoon_mesh_detailed(
                        &trace.ca[start..end],
                        &ss.assignment[start..end],
                        &trace.plddts[start..end],
                        &trace.guides[start..end],
                        10,
                        16,
                    );
                    for v in &mut part.vertices {
                        v.residue_index += start;
                    }
                    ribbon.merge(part);
                }
                start = end;
            }
        }

        let predicted = diff.candidate_confidence.is_predicted();
        let base_colors: Vec<ColorRGB> = (0..n)
            .map(|i| {
                if predicted {
                    plddt_to_color(trace.plddts[i] as f32)
                } else {
                    secondary_structure_to_color(
                        ss.assignment
                            .get(i)
                            .copied()
                            .unwrap_or(SecondaryStructure::Coil),
                    )
                }
            })
            .collect();

        let center64 = trace.ca.iter().sum::<Vector3<f64>>() / n as f64;
        let center = center64.cast::<f32>();
        let ca32: Vec<Vector3<f32>> = trace.ca.iter().map(|p| p.cast::<f32>()).collect();
        let bounding_radius = ca32
            .iter()
            .map(|p| (p - center).norm())
            .fold(0.0f32, f32::max)
            .max(5.0);
        let mut camera = OrbitCamera::oriented(center, bounding_radius, &ca32);
        camera.pitch = PITCH;

        let labels: Vec<(String, isize)> =
            trace.ids.iter().map(|(c, s, _)| (c.clone(), *s)).collect();
        let index = residue_index(&labels);

        // Candidate atoms per ribbon residue, for sticks.
        let mut atoms_of: HashMap<usize, Vec<Atom>> = HashMap::new();
        for chain in trace.protein.chains() {
            let cid = chain.id().trim().to_string();
            for r in chain.residues() {
                if let Some(&i) = index.get(&(cid.clone(), r.serial_number())) {
                    atoms_of.entry(i).or_insert_with(|| {
                        r.atoms()
                            .map(|a| Atom {
                                name: a.name().trim().to_uppercase(),
                                pos: Vector3::new(a.x() as f32, a.y() as f32, a.z() as f32),
                            })
                            .collect()
                    });
                }
            }
        }
        let names: Vec<String> = trace.names.clone();

        let (w, h) = (settings.width, settings.height);
        let short = w.min(h) as f32;
        // Pixels per Å at zoom 1, and the caption panel the subject must clear.
        let base_scale = 1.0 / camera.angstroms_per_pixel(w, h) as f32;
        let panel = caption_panel_height(h);
        let lift_for = |zoom: f32| (panel * 0.5) / (base_scale * zoom);

        let overview_zoom = 0.95;
        let overview = View {
            center,
            yaw: -SWEEP,
            zoom: overview_zoom,
            focus: 0.0,
            clip: bounding_radius * 4.0,
            lift: lift_for(overview_zoom),
        };

        let mut groups = Vec::new();
        // Groups worth a shot: those with a clear local change. When none has one, the first
        // group is still shown, so the film says where the change is and that nothing broke.
        let has_change = |g: &Vec<usize>| {
            diff.changes
                .iter()
                .any(|c| c.clear && c.local && c.nearest_site.is_some_and(|s| g.contains(&s)))
        };
        let mut filmed: Vec<&Vec<usize>> = diff.groups.iter().filter(|g| has_change(g)).collect();
        if filmed.is_empty() {
            filmed.extend(diff.groups.first());
        }
        for (gi, g) in filmed.into_iter().enumerate().take(settings.max_groups) {
            let site_residues: Vec<usize> = g
                .iter()
                .filter_map(|&s| {
                    let site = &diff.sites[s];
                    index
                        .get(&(site.candidate_chain_id.clone(), site.candidate_res_seq))
                        .copied()
                })
                .collect();
            let changes: Vec<&proteus_core::variant::InteractionChange> = diff
                .changes
                .iter()
                .filter(|c| c.clear && c.local && c.nearest_site.is_some_and(|s| g.contains(&s)))
                .collect();

            // Sticks: the sites, and every residue an interaction of this group touches.
            let mut extra: HashMap<usize, Vec<String>> = HashMap::new();
            for &r in &site_residues {
                extra.entry(r).or_default();
            }
            for c in &changes {
                for e in [&c.a, &c.b] {
                    if let Some(id) = &e.candidate_residue {
                        if let Some(&r) = index.get(id) {
                            let list = extra.entry(r).or_default();
                            if let Some(a) = &e.atom {
                                list.push(a.clone());
                            }
                        }
                    }
                }
            }
            let mut keys: Vec<usize> = extra.keys().copied().collect();
            keys.sort_unstable();
            let mut sticks = Vec::new();
            for r in keys {
                if let Some(all) = atoms_of.get(&r) {
                    let atoms = shown_atoms(&names[r], all, &extra[&r]);
                    let bonds = bonds_of(&atoms);
                    sticks.push(Sticks {
                        atoms,
                        bonds,
                        ghost: false,
                    });
                }
            }
            // Ghosts: the parent's atoms that the candidate no longer has, with the atoms they
            // bond to as anchors.
            for &s in g {
                let site = &diff.sites[s];
                let cand: Vec<&str> = site
                    .candidate_atoms
                    .iter()
                    .map(|(n, _)| n.as_str())
                    .collect();
                let parent: Vec<Atom> = site
                    .parent_atoms
                    .iter()
                    .map(|(n, p)| Atom {
                        name: n.clone(),
                        pos: v32(*p),
                    })
                    .collect();
                let removed: Vec<usize> = (0..parent.len())
                    .filter(|&i| !cand.contains(&parent[i].name.as_str()))
                    .collect();
                if removed.is_empty() {
                    continue;
                }
                let all_bonds = bonds_of(&parent);
                let mut keep: Vec<usize> = removed.clone();
                for &(i, j) in &all_bonds {
                    if removed.contains(&i) && !keep.contains(&j) {
                        keep.push(j);
                    }
                    if removed.contains(&j) && !keep.contains(&i) {
                        keep.push(i);
                    }
                }
                keep.sort_unstable();
                let atoms: Vec<Atom> = keep.iter().map(|&i| parent[i].clone()).collect();
                let bonds = bonds_of(&atoms)
                    .into_iter()
                    .filter(|&(i, j)| removed.contains(&keep[i]) || removed.contains(&keep[j]))
                    .collect();
                sticks.push(Sticks {
                    atoms,
                    bonds,
                    ghost: true,
                });
            }
            let lines: Vec<Line> = changes
                .iter()
                .map(|c| {
                    // A lost interaction is drawn where its atoms are now, so a partner that
                    // moved away shows the stretched gap; an atom the change removed has no
                    // "now" and keeps the parent's position, on its ghost.
                    let at = |e: &proteus_core::variant::Endpoint| match c.change {
                        Change::Lost => v32(e.other_position.unwrap_or(e.position)),
                        Change::Gained => v32(e.position),
                    };
                    Line {
                        a: at(&c.a),
                        b: at(&c.b),
                        lost: c.change == Change::Lost,
                    }
                })
                .collect();

            // Frame the group: its sticks, its lines and the site C-alphas.
            let mut pts: Vec<Vector3<f32>> = site_residues.iter().map(|&r| ca32[r]).collect();
            for s in &sticks {
                pts.extend(s.atoms.iter().map(|a| a.pos));
            }
            for l in &lines {
                pts.push(l.a);
                pts.push(l.b);
            }
            if pts.is_empty() {
                continue;
            }
            let gc = pts.iter().sum::<Vector3<f32>>() / pts.len() as f32;
            let radius = pts.iter().map(|p| (p - gc).norm()).fold(0.0f32, f32::max) + 3.0;
            let radius = radius.max(9.0);
            // Look at the group from outside the protein: rotate its direction from the
            // centre onto the viewer (+Z), in the camera's principal-axis frame.
            let dir = camera.base * (gc - center);
            let yaw = if dir.norm() > 1.0 {
                (-dir.x).atan2(dir.z)
            } else {
                0.0
            };
            let zoom = (0.34 * short / radius) / base_scale;
            let view = View {
                center: gc,
                yaw,
                zoom,
                focus: 1.0,
                clip: radius * 0.85,
                lift: lift_for(zoom),
            };

            let caption = group_caption(diff, g, &changes, gi);
            groups.push(Group {
                site_cas: site_residues.iter().map(|&r| ca32[r]).collect(),
                site_residues,
                sticks,
                lines,
                view,
                caption,
            });
        }

        let establish = establish_caption(diff);
        let summary = summary_caption(diff, &settings.title);
        let shown_groups = groups.len();
        let not_filmed = diff.groups.len().saturating_sub(shown_groups);
        let mut foot = Vec::new();
        if not_filmed > 0 {
            foot.push(format!("{not_filmed} site group(s) with no clear change"));
        }
        if diff.summary.marginal_local > 0 {
            foot.push(format!("{} marginal", diff.summary.marginal_local));
        }
        let distal = diff.summary.lost_distal.total() + diff.summary.gained_distal.total();
        if distal > 0 {
            foot.push(format!("{distal} distal"));
        }
        let footnote = if foot.is_empty() {
            String::new()
        } else {
            format!("not shown: {}", foot.join(" · "))
        };

        let secs = |s: f32| ((s * settings.pace * settings.fps as f32).round() as usize).max(1);
        let mut timeline = vec![(Segment::Establish, 0, secs(ESTABLISH_S))];
        let mut prev: Option<usize> = None;
        for g in 0..groups.len() {
            timeline.push((
                Segment::Fly {
                    from: prev,
                    to: Some(g),
                },
                0,
                secs(FLY_S),
            ));
            timeline.push((Segment::Hold(g), 0, secs(HOLD_S)));
            prev = Some(g);
        }
        if prev.is_some() {
            timeline.push((
                Segment::Fly {
                    from: prev,
                    to: None,
                },
                0,
                secs(FLY_S),
            ));
        }
        timeline.push((Segment::Summary, 0, secs(SUMMARY_S)));
        let mut at = 0;
        for seg in &mut timeline {
            seg.1 = at;
            at += seg.2;
        }

        Ok(Film {
            settings,
            ribbon,
            base_colors,
            camera,
            bounding_radius,
            overview,
            groups,
            timeline,
            frames: at,
            establish,
            summary,
            footnote,
        })
    }

    pub fn frame_count(&self) -> usize {
        self.frames
    }

    pub fn settings(&self) -> &FilmSettings {
        &self.settings
    }

    /// Site groups the film visits.
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// The frame used for the still: the middle of the first site's hold, or of the establish
    /// shot when nothing changed.
    /// The frame at `seconds` into the film (clamped to its length).
    pub fn frame_at(&self, seconds: f32) -> usize {
        ((seconds.max(0.0) * self.settings.fps as f32).round() as usize)
            .min(self.frames.saturating_sub(1))
    }

    pub fn still_frame(&self) -> usize {
        self.timeline
            .iter()
            .find(|(s, _, _)| matches!(s, Segment::Hold(_)))
            .or_else(|| self.timeline.first())
            .map(|&(_, start, len)| start + len / 2)
            .unwrap_or(0)
    }

    fn segment_at(&self, frame: usize) -> (Segment, f32, usize) {
        let frame = frame.min(self.frames.saturating_sub(1));
        for &(seg, start, len) in &self.timeline {
            if frame < start + len {
                // End-exclusive: a shot's last state is the next shot's first frame, so no
                // frame is shown twice at a cut. The film's final frame still reaches u = 1.
                let last = frame + 1 == self.frames;
                let u = if last {
                    1.0
                } else {
                    (frame - start) as f32 / len as f32
                };
                return (seg, u, frame - start);
            }
        }
        (Segment::Summary, 1.0, 0)
    }

    fn hold_view(&self, g: usize, u: f32) -> View {
        let mut v = self.groups[g].view;
        v.yaw += SWEEP * (2.0 * smoothstep(u) - 1.0) * 0.8;
        v
    }

    fn overview_at(&self, yaw_offset: f32) -> View {
        let mut v = self.overview;
        v.yaw = yaw_offset;
        v
    }

    /// Camera, colours and overlays at `frame`, plus how far each group's overlays have grown.
    fn view_at(&self, frame: usize) -> (View, Vec<(usize, f32)>) {
        let (seg, u, _) = self.segment_at(frame);
        match seg {
            Segment::Establish => (
                self.overview_at(SWEEP * (2.0 * smoothstep(u) - 1.0)),
                vec![],
            ),
            Segment::Hold(g) => (self.hold_view(g, u), vec![(g, 1.0)]),
            Segment::Fly { from, to } => {
                let t = smoothstep(u);
                let a = match from {
                    Some(g) => self.hold_view(g, 1.0),
                    None => self.overview_at(SWEEP),
                };
                let b = match to {
                    Some(g) => self.hold_view(g, 0.0),
                    None => self.overview_at(-SWEEP),
                };
                let mut grows = Vec::new();
                if let Some(g) = from {
                    grows.push((g, 1.0 - smoothstep(u * 1.6)));
                }
                if let Some(g) = to {
                    grows.push((g, smoothstep((u - 0.35) / 0.65)));
                }
                (View::mix(&a, &b, t), grows)
            }
            Segment::Summary => (self.overview_at(-SWEEP + SWEEP * smoothstep(u)), vec![]),
        }
    }

    /// The structure part of `frame`, for any renderer.
    pub fn scene(&self, frame: usize) -> FrameScene {
        let (view, grows) = self.view_at(frame);
        let mut camera = self.camera.clone();
        camera.center = view.center;
        camera.yaw = view.yaw;
        camera.pitch = PITCH;
        camera.zoom = view.zoom;
        camera.pan = Vector3::new(0.0, view.lift, 0.0);

        let ground = brand::DARK.ground;
        let mut site_glow = vec![0.0f32; self.base_colors.len()];
        for &(g, grow) in &grows {
            for &r in &self.groups[g].site_residues {
                site_glow[r] = site_glow[r].max(grow);
            }
        }
        // The establish and summary shots light every filmed site.
        let (seg, u, _) = self.segment_at(frame);
        let all_sites = match seg {
            // Lit from the first frame: feeds use frame 0 as the thumbnail.
            Segment::Establish => 1.0,
            Segment::Summary => smoothstep(u / 0.4),
            _ => 0.0,
        };
        if all_sites > 0.0 {
            for g in &self.groups {
                for &r in &g.site_residues {
                    site_glow[r] = site_glow[r].max(all_sites);
                }
            }
        }
        let residue_colors: Vec<ColorRGB> = self
            .base_colors
            .iter()
            .zip(&site_glow)
            .map(|(&c, &glow)| {
                let dimmed = ColorRGB::lerp(c, ColorRGB::lerp(c, ground, 0.80), view.focus);
                ColorRGB::lerp(dimmed, palette::SITE, glow)
            })
            .collect();

        let mut by_color: Vec<(TriangleMesh, ColorRGB)> = Vec::new();
        let mut push = |mesh: TriangleMesh, color: ColorRGB| {
            if let Some(slot) = by_color.iter_mut().find(|(_, c)| *c == color) {
                slot.0.merge(mesh);
            } else {
                by_color.push((mesh, color));
            }
        };
        if all_sites > 0.01 {
            for g in &self.groups {
                for &p in &g.site_cas {
                    push(
                        generate_sphere_mesh(
                            p,
                            1.6 * all_sites,
                            8,
                            14,
                            0,
                            SecondaryStructure::Coil,
                            90.0,
                        ),
                        palette::SITE,
                    );
                }
            }
        }
        for &(g, grow) in &grows {
            if grow <= 0.01 {
                continue;
            }
            let group = &self.groups[g];
            for s in &group.sticks {
                let r = if s.ghost { 0.13 } else { 0.24 } * grow;
                for &(i, j) in &s.bonds {
                    let (a, b) = (&s.atoms[i], &s.atoms[j]);
                    let mid = (a.pos + b.pos) * 0.5;
                    let (ca, cb) = if s.ghost {
                        (palette::GHOST, palette::GHOST)
                    } else {
                        (element_color(&a.name), element_color(&b.name))
                    };
                    let ss = SecondaryStructure::Coil;
                    push(generate_cylinder_mesh(a.pos, mid, r, 10, 0, ss, 90.0), ca);
                    push(generate_cylinder_mesh(mid, b.pos, r, 10, 0, ss, 90.0), cb);
                }
                for a in &s.atoms {
                    let c = if s.ghost {
                        palette::GHOST
                    } else {
                        element_color(&a.name)
                    };
                    push(
                        generate_sphere_mesh(
                            a.pos,
                            r * 1.25,
                            6,
                            10,
                            0,
                            SecondaryStructure::Coil,
                            90.0,
                        ),
                        c,
                    );
                }
            }
            for l in &group.lines {
                let color = if l.lost {
                    palette::LOST
                } else {
                    palette::GAINED
                };
                let r = 0.16 * grow.max(0.3);
                // The line draws itself from a to b as it grows.
                let end = l.a + (l.b - l.a) * grow;
                if l.lost {
                    let len = (end - l.a).norm();
                    let dir = (end - l.a) / len.max(1e-6);
                    let (dash, gap) = (0.36f32, 0.26f32);
                    let mut t = 0.0f32;
                    while t < len {
                        let t1 = (t + dash).min(len);
                        push(
                            generate_cylinder_mesh(
                                l.a + dir * t,
                                l.a + dir * t1,
                                r,
                                8,
                                0,
                                SecondaryStructure::Coil,
                                90.0,
                            ),
                            color,
                        );
                        t = t1 + gap;
                    }
                } else {
                    push(
                        generate_cylinder_mesh(l.a, end, r, 8, 0, SecondaryStructure::Coil, 90.0),
                        color,
                    );
                }
                for p in [l.a, end] {
                    push(
                        generate_sphere_mesh(p, r * 1.6, 5, 8, 0, SecondaryStructure::Coil, 90.0),
                        color,
                    );
                }
            }
        }

        let far = self.bounding_radius * 4.0;
        let clipping = view.clip < far * 0.99;
        let clip_near = clipping.then_some(-view.clip);
        // Fog over what is drawn: the whole protein, or from the slab to a little past the
        // group.
        let depth_range = (
            -view.clip.min(self.bounding_radius),
            lerp(self.bounding_radius, view.clip * 2.4, view.focus),
        );
        FrameScene {
            camera,
            residue_colors,
            overlays: by_color,
            clip_near,
            depth_range,
        }
    }

    /// Captions, legend and signature over a rendered frame.
    pub fn draw_overlay(&self, frame: usize, rgb: &mut [u8], text: &TextPainter) {
        let (w, h) = (self.settings.width, self.settings.height);
        let mut canvas = Canvas {
            rgb,
            width: w,
            height: h,
        };
        let s = w.min(h) as f32 / 1080.0;
        let theme = &brand::DARK;
        let (seg, u, k) = self.segment_at(frame);
        let fps = self.settings.fps as f32 * self.settings.pace;
        let fade_in = |k: usize| smoothstep(k as f32 / (0.35 * fps));

        // Top line: what is being compared, and the mark.
        let m = 48.0 * s;
        text.draw(
            &mut canvas,
            m,
            m + 22.0 * s,
            &self.settings.title,
            24.0 * s,
            Weight::Regular,
            theme.muted,
            1.0,
        );
        let mark = "proteus diff";
        let mw = text.measure(mark, 22.0 * s, Weight::SemiBold);
        text.draw(
            &mut canvas,
            w as f32 - m - mw,
            m + 22.0 * s,
            mark,
            22.0 * s,
            Weight::SemiBold,
            theme.accent,
            1.0,
        );

        let (caption, alpha) = match seg {
            Segment::Establish => (&self.establish, 1.0),
            Segment::Hold(g) => (&self.groups[g].caption, fade_in(k)),
            Segment::Fly { from, .. } => match from {
                Some(g) => (&self.groups[g].caption, 1.0 - smoothstep(u / 0.3)),
                None => (&self.establish, 1.0 - smoothstep(u / 0.3)),
            },
            Segment::Summary => (&self.summary, fade_in(k)),
        };
        if matches!(seg, Segment::Hold(_)) || matches!(seg, Segment::Fly { to: Some(_), .. }) {
            let la = match seg {
                Segment::Hold(_) => 1.0,
                _ => smoothstep((u - 0.5) / 0.5),
            };
            draw_legend(&mut canvas, text, s, la);
        }
        if alpha > 0.01 {
            draw_caption(&mut canvas, text, caption, s, alpha);
        }
        if matches!(seg, Segment::Summary) && !self.footnote.is_empty() {
            let panel = caption_panel_height(h);
            text.draw(
                &mut canvas,
                m,
                h as f32 - panel - 44.0 * s - 18.0 * s,
                &self.footnote,
                20.0 * s,
                Weight::Regular,
                theme.dim,
                fade_in(k),
            );
        }
    }

    /// Render `frame` on the CPU into `rgb` (`width × height × 3`), using `worker`'s buffers.
    pub fn render_cpu(&self, frame: usize, worker: &mut CpuWorker, rgb: &mut [u8]) {
        let scene = self.scene(frame);
        worker.render(self, &scene, rgb);
        self.draw_overlay(frame, rgb, &worker.text);
    }

    pub fn ribbon(&self) -> &TriangleMesh {
        &self.ribbon
    }

    /// Ribbon residues (the length of [`FrameScene::residue_colors`]).
    pub fn residue_count(&self) -> usize {
        self.base_colors.len()
    }
}

/// Height of the lower-third caption panel, in pixels.
fn caption_panel_height(h: usize) -> f32 {
    (h as f32 * 0.205).max(150.0)
}

fn draw_caption(canvas: &mut Canvas<'_>, text: &TextPainter, c: &Caption, s: f32, alpha: f32) {
    let theme = &brand::DARK;
    let (w, h) = (canvas.width as f32, canvas.height as f32);
    let m = 48.0 * s;
    let panel = caption_panel_height(canvas.height);
    let top = h - panel - 40.0 * s;
    canvas.fill_rect(
        m as usize,
        top as usize,
        (w - 2.0 * m) as usize,
        panel as usize,
        theme.surface,
        0.90 * alpha,
    );
    canvas.fill_rect(
        m as usize,
        top as usize,
        (6.0 * s).max(2.0) as usize,
        panel as usize,
        theme.accent,
        alpha,
    );
    let x = m + 34.0 * s;
    let mut y = top + 30.0 * s + text.ascent(44.0 * s);
    text.draw(
        canvas,
        x,
        y,
        &c.title,
        44.0 * s,
        Weight::SemiBold,
        theme.text,
        alpha,
    );
    y += 54.0 * s;
    // The second line shrinks to fit the panel rather than run off it.
    let max_w = w - 2.0 * m - 68.0 * s;
    let gap = 18.0 * s;
    let width_at = |px: f32| -> f32 {
        c.body
            .iter()
            .map(|(seg, _)| text.measure(seg, px, Weight::Regular))
            .sum::<f32>()
            + gap * c.body.len().saturating_sub(1) as f32
    };
    let mut body_px = 30.0 * s;
    while body_px > 16.0 * s && width_at(body_px) > max_w {
        body_px -= s;
    }
    let mut pen = x;
    for (seg, color) in &c.body {
        pen = text.draw(canvas, pen, y, seg, body_px, Weight::Regular, *color, alpha);
        pen += gap;
    }
    y += 44.0 * s;
    // The detail line is clipped to the panel rather than wrapped: it is a list of atoms.
    let detail = fit_words(&c.detail, max_w, |t| {
        text.measure(t, 22.0 * s, Weight::Regular)
    });
    text.draw(
        canvas,
        x,
        y,
        &detail,
        22.0 * s,
        Weight::Regular,
        theme.muted,
        alpha,
    );
}

/// `line` cut at a word boundary, with " …", so that `measure` of it is at most `max_w`.
fn fit_words(line: &str, max_w: f32, measure: impl Fn(&str) -> f32) -> String {
    if measure(line) <= max_w {
        return line.to_string();
    }
    let words: Vec<&str> = line.split(' ').collect();
    for n in (1..words.len()).rev() {
        let candidate = format!("{} …", words[..n].join(" ").trim_end());
        if measure(&candidate) <= max_w {
            return candidate;
        }
    }
    "…".to_string()
}

fn draw_legend(canvas: &mut Canvas<'_>, text: &TextPainter, s: f32, alpha: f32) {
    if alpha <= 0.01 {
        return;
    }
    let theme = &brand::DARK;
    let w = canvas.width as f32;
    let m = 48.0 * s;
    let y = m + 64.0 * s;
    let items: [(&str, ColorRGB, u8); 3] = [
        ("lost", palette::LOST, 0),
        ("gained", palette::GAINED, 1),
        ("parent", palette::GHOST, 2),
    ];
    let px = 20.0 * s;
    let sample = 34.0 * s;
    let mut total = 0.0;
    for (label, _, _) in &items {
        total += sample + 10.0 * s + text.measure(label, px, Weight::Regular) + 26.0 * s;
    }
    let mut x = w - m - total + 26.0 * s;
    for (label, color, style) in items {
        let yy = (y - px * 0.35) as usize;
        let thick = (3.0 * s).max(2.0) as usize;
        match style {
            0 => {
                let mut t = 0.0;
                while t < sample {
                    let d = (8.0 * s).min(sample - t);
                    canvas.fill_rect((x + t) as usize, yy, d as usize, thick, color, alpha);
                    t += 13.0 * s;
                }
            }
            1 => canvas.fill_rect(x as usize, yy, sample as usize, thick, color, alpha),
            _ => canvas.fill_rect(
                x as usize,
                yy,
                sample as usize,
                (thick / 2).max(1),
                color,
                alpha,
            ),
        }
        x += sample + 10.0 * s;
        x = text.draw(canvas, x, y, label, px, Weight::Regular, theme.muted, alpha) + 26.0 * s;
    }
}

fn kind_counts(
    changes: &[&proteus_core::variant::InteractionChange],
    ch: Change,
) -> Vec<(InteractionKind, usize)> {
    InteractionKind::ALL
        .iter()
        .map(|&k| {
            (
                k,
                changes
                    .iter()
                    .filter(|c| c.kind == k && c.change == ch)
                    .count(),
            )
        })
        .collect()
}

fn group_caption(
    diff: &VariantDiff,
    group: &[usize],
    changes: &[&proteus_core::variant::InteractionChange],
    gi: usize,
) -> Caption {
    let theme = &brand::DARK;
    let sites: Vec<&proteus_core::variant::ChangedSite> =
        group.iter().map(|&s| &diff.sites[s]).collect();
    let mut chains: Vec<&str> = sites
        .iter()
        .map(|s| s.candidate_chain_id.as_str())
        .collect();
    chains.dedup();
    let names: Vec<&str> = sites.iter().map(|s| s.notation.as_str()).collect();
    let title = if names.len() > 4 {
        format!("{}  +{} more", names[..4].join("  "), names.len() - 4)
    } else {
        names.join("  ")
    };
    let title = format!(
        "{title}   {}{}",
        if chains.len() == 1 {
            "chain "
        } else {
            "chains "
        },
        chains.join(",")
    );
    let lost = counts_phrase(&kind_counts(changes, Change::Lost), '−');
    let gained = counts_phrase(&kind_counts(changes, Change::Gained), '+');
    let interface = changes.iter().filter(|c| c.interface).count();
    let mut body = Vec::new();
    if !lost.is_empty() {
        body.push((lost, palette::LOST));
    }
    if !gained.is_empty() {
        body.push((gained, palette::GAINED));
    }
    if body.is_empty() {
        body.push(("no clear interaction change here".into(), theme.muted));
    } else if interface == changes.len() {
        body.push(("all at the interface".into(), theme.muted));
    } else if interface > 0 {
        body.push((format!("· {interface} at the interface"), theme.muted));
    }
    let confidences: Vec<String> = sites
        .iter()
        .filter_map(|s| {
            Some(format!(
                "{} pLDDT {:.0} → {:.0}",
                s.notation, s.parent_confidence?, s.candidate_confidence?
            ))
        })
        .collect();
    let detail = if !confidences.is_empty() {
        confidences.join(" · ")
    } else {
        let mut removed: Vec<String> = changes
            .iter()
            .flat_map(|c| c.removed.iter().cloned())
            .collect();
        removed.sort();
        removed.dedup();
        if removed.is_empty() {
            let rmsd: Vec<String> = sites
                .iter()
                .map(|s| format!("{} local C-alpha RMSD {:.2} Å", s.notation, s.local_ca_rmsd))
                .collect();
            rmsd.join(" · ")
        } else {
            format!("removed: {}", removed.join(", "))
        }
    };
    let _ = gi;
    Caption {
        title,
        body,
        detail,
    }
}

fn establish_caption(diff: &VariantDiff) -> Caption {
    let theme = &brand::DARK;
    let n = diff.sites.len();
    let title = match n {
        0 => "no residue changed".to_string(),
        1 => "1 changed residue".to_string(),
        _ => format!("{n} changed residues"),
    };
    let body = vec![(
        format!(
            "C-alpha RMSD {:.2} Å over {} paired residues",
            diff.global_ca_rmsd, diff.paired
        ),
        theme.text,
    )];
    // Residues with no partner are not "changed sites", but a reader comparing with the
    // PDB's mutation list needs to know they exist (1X1Y's Q2A is unmodelled).
    let mut unpaired = Vec::new();
    if diff.deleted > 0 {
        unpaired.push(format!("{} parent", diff.deleted));
    }
    if diff.inserted > 0 {
        unpaired.push(format!("{} new", diff.inserted));
    }

    let detail = if diff.candidate_confidence.is_predicted() {
        match (
            diff.summary.min_site_confidence,
            diff.summary.mean_site_confidence_change,
        ) {
            (Some(min), Some(delta)) => format!(
                "coloured by pLDDT · lowest at a site {min:.0} · mean change at sites {delta:+.1}"
            ),
            (Some(min), None) => format!("coloured by pLDDT · lowest at a site {min:.0}"),
            _ => "coloured by pLDDT".into(),
        }
    } else {
        "experimental: no model confidence".into()
    };
    let detail = if unpaired.is_empty() {
        detail
    } else {
        format!("{detail} · unpaired: {}", unpaired.join(", "))
    };
    Caption {
        title,
        body,
        detail,
    }
}

fn summary_caption(diff: &VariantDiff, title: &str) -> Caption {
    let theme = &brand::DARK;
    let s = &diff.summary;
    let lost: Vec<(InteractionKind, usize)> = InteractionKind::ALL
        .iter()
        .map(|&k| (k, s.lost_local.get(k)))
        .collect();
    let gained: Vec<(InteractionKind, usize)> = InteractionKind::ALL
        .iter()
        .map(|&k| (k, s.gained_local.get(k)))
        .collect();
    let mut body = Vec::new();
    let (l, g) = (counts_phrase(&lost, '−'), counts_phrase(&gained, '+'));
    if !l.is_empty() {
        body.push((l, palette::LOST));
    }
    if !g.is_empty() {
        body.push((g, palette::GAINED));
    }
    if body.is_empty() {
        body.push(("no clear local interaction change".into(), theme.muted));
    }
    let interface = s.interface_lost_local + s.interface_gained_local;
    let changed = diff.summary.lost_local.total() + diff.summary.gained_local.total();
    let sites = match diff.sites.len() {
        1 => "1 changed residue".to_string(),
        n => format!("{n} changed residues"),
    };
    let detail = format!("{sites} · {interface} of {changed} changes at the interface");
    Caption {
        title: if title.is_empty() {
            "summary".into()
        } else {
            title.to_string()
        },
        body,
        detail,
    }
}

/// Per-thread buffers for CPU rendering.
pub struct CpuWorker {
    hi: Framebuffer,
    lo: Framebuffer,
    raster: Rasterizer,
    pub text: TextPainter,
}

impl Default for CpuWorker {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuWorker {
    pub fn new() -> Self {
        Self {
            hi: Framebuffer::new(1, 1),
            lo: Framebuffer::new(1, 1),
            raster: Rasterizer::new(ColorScheme::Scores),
            text: TextPainter::new(),
        }
    }

    /// Draw `scene`'s structure into `rgb` on the ground colour.
    pub fn render(&mut self, film: &Film, scene: &FrameScene, rgb: &mut [u8]) {
        let (w, h) = (film.settings.width, film.settings.height);
        let k = film.settings.supersample.max(1);
        self.hi.resize(w * k, h * k);
        let ground = brand::DARK.ground;
        self.hi.clear(ground);
        let r = &mut self.raster;
        r.residue_colors = scene.residue_colors.clone();
        r.clip_near = scene.clip_near;
        r.depth_range = Some(scene.depth_range);
        r.rasterize_mesh(
            &film.ribbon,
            &scene.camera,
            &mut self.hi,
            ColorScheme::Scores,
        );
        for (mesh, color) in &scene.overlays {
            r.rasterize_mesh(
                mesh,
                &scene.camera,
                &mut self.hi,
                ColorScheme::Solid(*color),
            );
        }
        downsample_on_ground(&self.hi, &mut self.lo, k, ground);
        r.apply_post_processing(&mut self.lo);
        for (i, c) in self.lo.colors.iter().enumerate() {
            rgb[i * 3] = c.r;
            rgb[i * 3 + 1] = c.g;
            rgb[i * 3 + 2] = c.b;
        }
    }
}

/// Box-filter a `k`× supersampled buffer, blending edges with the ground colour (a film has a
/// background, unlike a terminal, so edges should be antialiased against it).
fn downsample_on_ground(hi: &Framebuffer, lo: &mut Framebuffer, k: usize, ground: ColorRGB) {
    let (w, h) = (hi.width / k, hi.height / k);
    lo.resize(w, h);
    let kk = (k * k) as u32;
    for y in 0..h {
        for x in 0..w {
            let (mut r, mut g, mut b) = (0u32, 0u32, 0u32);
            let mut depth = f32::INFINITY;
            let mut hits = 0;
            for sy in 0..k {
                let row = (y * k + sy) * hi.width + x * k;
                for sx in 0..k {
                    let i = row + sx;
                    let c = if hi.depths[i].is_finite() {
                        hits += 1;
                        depth = depth.min(hi.depths[i]);
                        hi.colors[i]
                    } else {
                        ground
                    };
                    r += c.r as u32;
                    g += c.g as u32;
                    b += c.b as u32;
                }
            }
            let o = y * w + x;
            lo.colors[o] = ColorRGB::new((r / kk) as u8, (g / kk) as u8, (b / kk) as u8);
            // A pixel mostly empty stays "empty" for the outline and SSAO pass.
            lo.depths[o] = if hits * 2 >= k * k {
                depth
            } else {
                f32::INFINITY
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proteus_core::variant::{diff, DiffOptions, PreparedStructure};

    const PTC: &str = include_str!("../../../proteus-core/tests/data/2ptc_EI.pdb");

    /// BPTI Lys15 truncated to alanine, as in `proteus_core::variant`'s tests.
    fn k15a() -> String {
        let mut out = String::new();
        for line in PTC.lines() {
            let hit = line.starts_with("ATOM")
                && line.len() > 26
                && line.as_bytes()[21] == b'I'
                && line[22..26].trim() == "15";
            if hit {
                if !["N", "CA", "C", "O", "CB"].contains(&line[12..16].trim()) {
                    continue;
                }
                out.push_str(&format!("{}ALA{}\n", &line[..17], &line[20..]));
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
        out
    }

    fn small_film() -> (Film, VariantDiff) {
        let opts = DiffOptions::default();
        let parent = PreparedStructure::from_bytes(PTC.as_bytes(), Some("p.pdb"), None).unwrap();
        let cand_text = k15a();
        let cand =
            PreparedStructure::from_bytes(cand_text.as_bytes(), Some("c.pdb"), None).unwrap();
        let d = diff(&parent, &cand, &opts).unwrap();
        let settings = FilmSettings {
            width: 160,
            height: 160,
            fps: 6,
            supersample: 1,
            title: "k15a vs 2ptc".into(),
            ..FilmSettings::default()
        };
        let film = Film::new(cand_text.as_bytes(), Some("c.pdb"), None, &d, settings).unwrap();
        (film, d)
    }

    #[test]
    fn frames_are_the_same_bytes_every_time() {
        let (film, _) = small_film();
        assert_eq!(film.group_count(), 1);
        let n = 160 * 160 * 3;
        for frame in [0, film.still_frame(), film.frame_count() - 1] {
            let (mut a, mut b) = (vec![0u8; n], vec![0u8; n]);
            film.render_cpu(frame, &mut CpuWorker::new(), &mut a);
            film.render_cpu(frame, &mut CpuWorker::new(), &mut b);
            assert!(a == b, "frame {frame} differs between runs");
            assert!(a.iter().any(|&v| v != a[0]), "frame {frame} is blank");
        }
    }

    /// The GPU path against the CPU reference on the same frames. Not bit-identical (rasterisation
    /// rules and rounding differ by vendor), so: mean difference small, and nearly every pixel
    /// close. Skipped where there is no hardware adapter (CI runners).
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_frames_match_the_cpu_reference() {
        let (film, _) = small_film();
        let mut gpu = match gpu::GpuRenderer::new(&film) {
            Ok(g) => g,
            Err(why) => {
                eprintln!("skipped: {why}");
                return;
            }
        };
        let text = TextPainter::new();
        let n = 160 * 160 * 3;
        for frame in [0, film.still_frame(), film.frame_count() - 1] {
            let (mut cpu, mut gpu_rgb) = (vec![0u8; n], vec![0u8; n]);
            film.render_cpu(frame, &mut CpuWorker::new(), &mut cpu);
            film.render_gpu(frame, &mut gpu, &text, &mut gpu_rgb)
                .unwrap();
            let diffs: Vec<u32> = cpu
                .iter()
                .zip(&gpu_rgb)
                .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
                .collect();
            let mean = diffs.iter().sum::<u32>() as f64 / n as f64;
            let close = diffs
                .chunks(3)
                .filter(|p| p.iter().all(|&d| d <= 24))
                .count() as f64
                / (n / 3) as f64;
            eprintln!(
                "frame {frame}: mean |Δ| {mean:.2}, {:.1} % of pixels within 24",
                close * 100.0
            );
            assert!(mean < 4.0, "frame {frame}: mean difference {mean:.2}");
            assert!(
                close > 0.95,
                "frame {frame}: only {:.1} % of pixels close",
                close * 100.0
            );
        }
    }

    /// Consecutive frames never repeat the camera: a repeated frame at a cut reads as a hitch.
    #[test]
    fn no_frame_repeats_the_one_before_it() {
        let (film, _) = small_film();
        let state = |f: usize| {
            let c = film.scene(f).camera;
            (c.center, c.yaw, c.zoom, c.pan)
        };
        for f in 1..film.frame_count() {
            assert!(
                state(f) != state(f - 1),
                "frame {f} repeats frame {}",
                f - 1
            );
        }
    }

    /// Every number a caption shows is the diff's own count.
    #[test]
    fn captions_carry_the_diffs_numbers() {
        let (film, d) = small_film();
        let caption = &film.groups[0].caption;
        assert!(caption.title.starts_with("K15A"), "{}", caption.title);
        let body: String = caption.body.iter().map(|(t, _)| t.as_str()).collect();
        for kind in InteractionKind::ALL {
            let n = d
                .changes
                .iter()
                .filter(|c| c.clear && c.local && c.kind == kind && c.change == Change::Lost)
                .count();
            if n > 0 {
                let phrase = format!("−{n} {}", kind.label(n));
                assert!(body.contains(&phrase), "{body:?} lacks {phrase:?}");
            }
        }
        let summary: String = film.summary.body.iter().map(|(t, _)| t.as_str()).collect();
        let hb = d.summary.lost_local.hydrogen_bonds;
        if hb > 0 {
            assert!(summary.contains(&format!("−{hb} H-bond")), "{summary}");
        }
    }

    /// A line one glyph wider than its box, whose cut ends in a space, once looped forever.
    #[test]
    fn fit_words_terminates_and_fits() {
        let measure = |t: &str| t.chars().count() as f32;
        assert_eq!(fit_words("short", 10.0, measure), "short");
        let cut = fit_words("4 changed residues · 5 at the interface", 20.0, measure);
        assert!(measure(&cut) <= 20.0, "{cut}");
        assert!(cut.ends_with(" …"));
        assert_eq!(fit_words("unbreakableword", 3.0, measure), "…");
    }
}
