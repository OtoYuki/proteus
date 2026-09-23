use crate::error::RenderError;
use crate::geometry::mesh::TriangleMesh;
use crate::rasterizer::buffer::{ColorRGB, Framebuffer};
use crate::rasterizer::camera::OrbitCamera;
use crate::rasterizer::pipeline::Rasterizer;
use crate::rasterizer::shader::ColorScheme;
use crate::terminal::halfblock::HalfBlockRenderer;
use crate::tui::dashboard::{fit_to_width, DashboardData, DashboardRenderer};
use crossterm::cursor::{Hide, Show};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use std::fmt::Write as _;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Put the terminal back the way the shell expects it: cooked mode, main screen, cursor shown.
fn restore_terminal() {
    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), LeaveAlternateScreen, Show);
}

struct RawTerminalGuard;

impl Drop for RawTerminalGuard {
    fn drop(&mut self) {
        restore_terminal();
    }
}

type PanicHook = Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Sync + Send + 'static>;

/// While alive, a panic first runs `restore` and then the previous hook. Without it a panic
/// inside the viewer prints its message into the alternate screen, which is then discarded,
/// and leaves the shell in raw mode with no cursor. Dropping it reinstates the previous hook.
struct PanicHookGuard {
    previous: Option<Arc<PanicHook>>,
    active: Arc<AtomicBool>,
}

impl PanicHookGuard {
    fn install(restore: fn()) -> Self {
        let previous: Arc<PanicHook> = Arc::new(std::panic::take_hook());
        let active = Arc::new(AtomicBool::new(true));
        let (chained, armed) = (Arc::clone(&previous), Arc::clone(&active));
        std::panic::set_hook(Box::new(move |info| {
            if armed.load(Ordering::SeqCst) {
                restore();
            }
            chained(info);
        }));
        Self {
            previous: Some(previous),
            active,
        }
    }
}

impl Drop for PanicHookGuard {
    fn drop(&mut self) {
        self.active.store(false, Ordering::SeqCst);
        // `set_hook` itself panics on a panicking thread, which during unwinding would abort;
        // the disarmed hook stays installed then and only forwards to the previous one.
        if std::thread::panicking() {
            return;
        }
        if let Some(previous) = self.previous.take() {
            std::panic::set_hook(Box::new(move |info| previous(info)));
        }
    }
}

/// Terminal width below which the dashboard is not drawn beside the view.
pub const DASHBOARD_MIN_COLS: u16 = 90;
/// Fewest rows the dashboard can be drawn in (see [`DashboardRenderer::render_to_buffer`]).
pub const DASHBOARD_MIN_ROWS: u16 = 10;

/// Where each part of the interactive screen goes on a terminal of `cols` × `rows` cells.
///
/// Every region lies inside the terminal and none overlaps another, for any size including
/// zero: the view takes what the HUD leaves, the HUD shrinks to one line and then none on a
/// very short terminal, and the dashboard is hidden rather than squeezed when it does not fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewerLayout {
    pub cols: u16,
    pub rows: u16,
    /// Cells given to the 3-D view, anchored top-left.
    pub view_cols: u16,
    pub view_rows: u16,
    /// Dashboard panel as (first column, width), 0-based; the separator is the column before.
    pub dashboard: Option<(u16, u16)>,
    /// Status lines under the view: 2, or fewer on a short terminal.
    pub hud_rows: u16,
}

impl ViewerLayout {
    pub fn compute(cols: u16, rows: u16, dashboard: bool) -> Self {
        let hud_rows = match rows {
            0..=3 => 0,
            4..=7 => 1,
            _ => 2,
        };
        let view_rows = rows - hud_rows;
        let (view_cols, dashboard) =
            if dashboard && cols >= DASHBOARD_MIN_COLS && view_rows >= DASHBOARD_MIN_ROWS {
                let left = (u32::from(cols) * 58 / 100) as u16;
                (left, Some((left + 1, cols - left - 1)))
            } else {
                (cols, None)
            };
        Self {
            cols,
            rows,
            view_cols,
            view_rows,
            dashboard,
            hud_rows,
        }
    }

    /// The HUD as positioned lines, each exactly the terminal width so that nothing wraps: a
    /// line one column too wide on the bottom row scrolls the whole screen every frame.
    /// `status` is the state line, `controls` the key help, both already styled; with one HUD
    /// row only `status` is shown.
    pub fn hud_lines(&self, status: &str, controls: &str) -> Vec<String> {
        let width = self.cols as usize;
        let mut lines: Vec<&str> = Vec::new();
        if self.hud_rows >= 1 {
            lines.push(status);
        }
        if self.hud_rows >= 2 {
            lines.push(controls);
        }
        lines
            .into_iter()
            .enumerate()
            .map(|(i, text)| {
                let row = self.view_rows as usize + i + 1;
                format!("\x1b[{row};1H{}\x1b[0m", fit_to_width(text, width))
            })
            .collect()
    }
}

pub struct ViewerConfig {
    pub title: String,
    pub initial_color_scheme: ColorScheme,
    pub auto_rotate: bool,
    pub secondary_mesh: Option<(TriangleMesh, ColorRGB)>,
    pub rmsd: Option<f64>,
    pub disulfide_mesh: Option<TriangleMesh>,
    pub dashboard_enabled: bool,
    pub dashboard_data: Option<DashboardData>,
    /// Set from elsewhere (a SIGTERM/SIGHUP handler) to make the viewer restore the terminal
    /// and return. Raw mode turns Ctrl-C into a key press, but a signal still terminates the
    /// process outright, leaving the terminal raw and on the alternate screen.
    pub stop: Option<Arc<AtomicBool>>,
}

impl Default for ViewerConfig {
    fn default() -> Self {
        Self {
            title: "Proteus 3D Viewer".to_string(),
            initial_color_scheme: ColorScheme::Plddt,
            auto_rotate: true,
            secondary_mesh: None,
            rmsd: None,
            disulfide_mesh: None,
            dashboard_enabled: false,
            dashboard_data: None,
            stop: None,
        }
    }
}

/// Run the interactive terminal ribbon viewer (frames are paced at 16 ms; the achieved rate is
/// shown in the HUD and depends on the terminal and structure size).
pub fn run_interactive_viewer(
    mesh: &TriangleMesh,
    mut camera: OrbitCamera,
    config: ViewerConfig,
) -> Result<(), RenderError> {
    // Declared first so it is dropped last, after the terminal has been restored.
    let _panic_guard = PanicHookGuard::install(restore_terminal);
    enable_raw_mode().map_err(|e| RenderError::Terminal(e.to_string()))?;
    let _guard = RawTerminalGuard;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, Hide)
        .map_err(|e| RenderError::Terminal(e.to_string()))?;

    let (term_cols, term_rows) =
        crossterm::terminal::size().map_err(|e| RenderError::Terminal(e.to_string()))?;

    let mut dashboard_mode = config.dashboard_enabled && config.dashboard_data.is_some();
    let dashboard_renderer = DashboardRenderer::new();
    use crate::brand::Role;

    let mut layout = ViewerLayout::compute(term_cols, term_rows, dashboard_mode);
    // Halfblock resolution: 1 character row = 2 pixel rows
    let mut fb = Framebuffer::new(layout.view_cols as usize, layout.view_rows as usize * 2);

    let mut rasterizer = Rasterizer::new(config.initial_color_scheme);
    let ansi = crate::brand::ansi::Ansi::detect();
    let mut compositor = HalfBlockRenderer::new();
    let mut auto_rotate = config.auto_rotate;
    let mut color_scheme = config.initial_color_scheme;
    let mut show_disulfides = config.disulfide_mesh.is_some();

    let mut out_buf = String::with_capacity(64 * 1024);
    let mut last_frame = Instant::now();
    let mut fps = 0.0f32;
    let mut frame_count = 0usize;
    let mut fps_timer = Instant::now();

    loop {
        if config
            .stop
            .as_ref()
            .is_some_and(|s| s.load(Ordering::Relaxed))
        {
            break;
        }

        // Handle input events
        let timeout = Duration::from_millis(16);
        let mut relayout: Option<(u16, u16)> = None;
        if event::poll(timeout).map_err(|e| RenderError::Terminal(e.to_string()))? {
            match event::read().map_err(|e| RenderError::Terminal(e.to_string()))? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                    KeyCode::Char(' ') => auto_rotate = !auto_rotate,
                    KeyCode::Tab | KeyCode::Char('b') => {
                        if config.dashboard_data.is_some() {
                            dashboard_mode = !dashboard_mode;
                            relayout = Some((layout.cols, layout.rows));
                        }
                    }
                    KeyCode::Char('c') => {
                        color_scheme = match color_scheme {
                            ColorScheme::Plddt => ColorScheme::SecondaryStructure,
                            ColorScheme::SecondaryStructure => ColorScheme::Rainbow,
                            ColorScheme::Rainbow => ColorScheme::Plddt,
                            ColorScheme::Solid(_) => ColorScheme::Plddt,
                        };
                        rasterizer.color_scheme = color_scheme;
                    }
                    KeyCode::Char('o') => {
                        let current = rasterizer.enable_ssao && rasterizer.enable_outlines;
                        rasterizer.enable_ssao = !current;
                        rasterizer.enable_outlines = !current;
                    }
                    KeyCode::Char('d') => {
                        show_disulfides = !show_disulfides;
                    }
                    KeyCode::Char('r') => {
                        camera.reset();
                    }
                    KeyCode::Left | KeyCode::Char('h') => camera.rotate(-0.15, 0.0),
                    KeyCode::Right | KeyCode::Char('l') => camera.rotate(0.15, 0.0),
                    KeyCode::Up | KeyCode::Char('k') => camera.rotate(0.0, 0.15),
                    KeyCode::Down | KeyCode::Char('j') => camera.rotate(0.0, -0.15),
                    KeyCode::Char('+') | KeyCode::Char('=') => camera.adjust_zoom(1.15),
                    KeyCode::Char('-') | KeyCode::Char('_') => camera.adjust_zoom(0.85),
                    _ => {}
                },
                Event::Resize(new_cols, new_rows) => relayout = Some((new_cols, new_rows)),
                _ => {}
            }
        }

        // A resize or a dashboard toggle moves every region, not only the view: recompute the
        // whole layout, and repaint from scratch because the screen is cleared.
        if let Some((cols, rows)) = relayout {
            layout = ViewerLayout::compute(cols, rows, dashboard_mode);
            fb.resize(layout.view_cols as usize, layout.view_rows as usize * 2);
            compositor.invalidate();
            let _ = execute!(
                stdout,
                crossterm::terminal::Clear(crossterm::terminal::ClearType::All)
            );
        }

        // Auto spin around Y axis
        let dt = last_frame.elapsed().as_secs_f32();
        last_frame = Instant::now();
        if auto_rotate {
            camera.rotate(0.75 * dt, 0.0);
        }

        // FPS calculation
        frame_count += 1;
        if fps_timer.elapsed() >= Duration::from_millis(500) {
            fps = frame_count as f32 / fps_timer.elapsed().as_secs_f32();
            frame_count = 0;
            fps_timer = Instant::now();
        }

        out_buf.clear();
        if layout.view_cols > 0 && layout.view_rows > 0 {
            // Render frame
            fb.clear(ColorRGB::BLACK); // black = empty: the terminal's own background shows
            rasterizer.rasterize_mesh(mesh, &camera, &mut fb, color_scheme);

            // Render superimposed secondary mesh if present
            if let Some((ref sec_mesh, sec_color)) = config.secondary_mesh {
                rasterizer.rasterize_mesh(
                    sec_mesh,
                    &camera,
                    &mut fb,
                    ColorScheme::Solid(sec_color),
                );
            }

            // Render disulfide bridges if present and enabled
            if show_disulfides {
                if let Some(ref ds_mesh) = config.disulfide_mesh {
                    let gold = crate::brand::structure::DISULFIDE;
                    rasterizer.rasterize_mesh(ds_mesh, &camera, &mut fb, ColorScheme::Solid(gold));
                }
            }

            // Post-processing: Screen-space ambient occlusion + cartoon silhouette outlines
            rasterizer.apply_post_processing(&mut fb);

            // Compose to terminal
            compositor.render_differential(&fb, &mut out_buf, 0, 0);
        }

        // Render the side-by-side biophysical dashboard where the layout has room for it
        if let (Some((dash_col, dash_width)), Some(d_data)) =
            (layout.dashboard, config.dashboard_data.as_ref())
        {
            // Vertical separator in the column just left of the panel (1-based `dash_col`)
            for r in 0..layout.view_rows {
                let row_pos = r + 1;
                let _ = write!(
                    out_buf,
                    "\x1b[{row_pos};{dash_col}H{}",
                    ansi.paint(Role::Line, "│")
                );
            }
            dashboard_renderer.render_to_buffer(
                d_data,
                &mut out_buf,
                dash_col,
                0,
                dash_width as usize,
                layout.view_rows as usize,
            );
        }

        // The status bar: the structure, a legend for the current colours (swatch and word, so
        // it reads without colour), the toggles, the frame rate; then the keys.
        let sep = ansi.paint(Role::Line, "  ·  ");
        let sw = |c: ColorRGB, word: &str| {
            format!(
                "{} {}",
                ansi.paint_rgb(c, "■"),
                ansi.paint(Role::Text, word)
            )
        };
        // Unknown (no analysis) counts as predicted: the pLDDT scheme is only the default then.
        let predicted = config
            .dashboard_data
            .as_ref()
            .and_then(|d| d.metrics.as_ref())
            .is_none_or(|m| m.plddt().is_some());
        let legend = if let Some(rmsd) = config.rmsd {
            format!(
                "{} {}  {}",
                sw(crate::brand::structure::TARGET, "target"),
                sw(crate::brand::structure::REFERENCE, "reference"),
                ansi.paint(Role::Text, &format!("RMSD {rmsd:.3} Å"))
            )
        } else {
            match color_scheme {
                ColorScheme::SecondaryStructure => [
                    sw(crate::brand::structure::HELIX, "helix"),
                    sw(crate::brand::structure::STRAND, "strand"),
                    sw(crate::brand::structure::COIL, "coil"),
                ]
                .join("  "),
                ColorScheme::Plddt => {
                    let s = crate::rasterizer::shader::plddt_to_color;
                    format!(
                        "{}  {}",
                        [
                            sw(s(95.0), ">90"),
                            sw(s(80.0), "70–90"),
                            sw(s(60.0), "50–70"),
                            sw(s(25.0), "<50")
                        ]
                        .join(" "),
                        // On an experimental file this is the B-factor column drawn on the
                        // pLDDT scale, and the legend says so.
                        if predicted {
                            ansi.paint(Role::Dim, "pLDDT")
                        } else {
                            ansi.paint(
                                Role::Warm,
                                "! B-factor on the pLDDT scale, not a confidence",
                            )
                        }
                    )
                }
                ColorScheme::Rainbow => ansi.paint(Role::Text, "rainbow, N → C"),
                ColorScheme::Solid(c) => sw(c, "solid"),
            }
        };
        let toggle = |name: &str, state: Option<bool>| match state {
            Some(true) => format!(
                "{} {}",
                ansi.paint(Role::Dim, name),
                ansi.paint(Role::Text, "on")
            ),
            Some(false) => format!(
                "{} {}",
                ansi.paint(Role::Dim, name),
                ansi.paint(Role::Dim, "off")
            ),
            None => String::new(),
        };
        let fx_on = rasterizer.enable_ssao && rasterizer.enable_outlines;
        let dash_state = config.dashboard_data.as_ref().map(|_| dashboard_mode);
        let dash_note =
            if dashboard_mode && config.dashboard_data.is_some() && layout.dashboard.is_none() {
                ansi.paint(Role::Warm, " (no room)")
            } else {
                String::new()
            };
        let parts: Vec<String> = [
            format!(
                "{} {}",
                ansi.paint(Role::Accent, "proteus"),
                ansi.paint(Role::Text, &config.title)
            ),
            legend,
            toggle(
                "disulfides",
                config.disulfide_mesh.as_ref().map(|_| show_disulfides),
            ),
            format!("{}{dash_note}", toggle("dashboard", dash_state)),
            toggle("effects", Some(fx_on)),
            toggle("spin", Some(auto_rotate)),
            ansi.paint(Role::Dim, &format!("{fps:.0} fps")),
        ]
        .into_iter()
        .filter(|p| !p.is_empty())
        .collect();
        let status_row1 = format!(" {}", parts.join(&sep));
        let key = |k: &str, v: &str| {
            format!(
                "{} {}",
                ansi.paint(Role::Accent, k),
                ansi.paint(Role::Dim, v)
            )
        };
        let status_row2 = format!(
            " {}",
            [
                key("←↑↓→ hjkl", "orbit"),
                key("+ −", "zoom"),
                key("space", "spin"),
                key("tab", "dashboard"),
                key("c", "colour"),
                key("o", "effects"),
                key("d", "disulfides"),
                key("r", "reset"),
                key("q", "back"),
            ]
            .join(&sep)
        );
        let status_row2 = status_row2.as_str();

        for line in layout.hud_lines(&status_row1, status_row2) {
            out_buf.push_str(&line);
        }

        if !out_buf.is_empty() {
            let _ = stdout.write_all(out_buf.as_bytes());
            let _ = stdout.flush();
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::dashboard::visible_width;
    use std::sync::atomic::AtomicUsize;

    /// The status line as the viewer formats it for a real file, and the key help line,
    /// which alone is 118 columns.
    const STATUS: &str = " 1crn.pdb | Tris: 3520 | Color: Secondary Structure | S-S: ON  | Dash: OFF | FX: ON  | Spin: ON  | 60 FPS";
    const CONTROLS: &str = " [arrows/hjkl] Orbit | [+/-] Zoom | [Space] Spin | [Tab] Dashboard | [c] Color | [o] FX | [d] S-S | [r] Reset | [q] Quit";

    /// At 80×24 the HUD used to be padded but never cut, so both lines wrapped, the bottom row
    /// scrolled the screen every frame, and the protein scrolled away under copies of the
    /// status line. Every HUD line must be exactly the terminal width, in display columns.
    #[test]
    fn hud_lines_never_exceed_the_terminal_width() {
        let wide_title = " 蛋白質_model_ünïcödé_φψ.cif | Superimposed RMSD: 0.123 Å | Tris: 1";
        for cols in 0..=200u16 {
            for rows in [0u16, 1, 3, 4, 7, 8, 24, 60] {
                let layout = ViewerLayout::compute(cols, rows, false);
                for status in [STATUS, wide_title] {
                    let lines = layout.hud_lines(status, CONTROLS);
                    assert_eq!(lines.len(), layout.hud_rows as usize);
                    for line in &lines {
                        assert_eq!(
                            visible_width(line),
                            cols as usize,
                            "{cols}x{rows}: HUD line is not the terminal width: {line:?}"
                        );
                    }
                }
            }
        }
    }

    /// The layout must place every region on screen without overlap, for any terminal size,
    /// and follow a resize. The old code fixed the view at `max(rows - 2, 10)` rows, so on a
    /// terminal under 12 rows the view and dashboard overdrew the HUD (and ran off-screen).
    #[test]
    fn layout_regions_fit_and_never_overlap() {
        for rows in 0..=70u16 {
            for cols in (0..=260u16).step_by(7).chain([89, 90, 91, 1200, u16::MAX]) {
                for dash in [false, true] {
                    let l = ViewerLayout::compute(cols, rows, dash);
                    assert_eq!(
                        l.view_rows + l.hud_rows,
                        rows,
                        "{cols}x{rows}: view and HUD do not tile the height"
                    );
                    assert!(l.view_cols <= cols);
                    match l.dashboard {
                        Some((first, width)) => {
                            assert!(dash, "{cols}x{rows}: dashboard shown while off");
                            // Separator at `first - 1`, directly right of the view.
                            assert_eq!(first, l.view_cols + 1, "{cols}x{rows}");
                            assert_eq!(first as u32 + width as u32, cols as u32, "{cols}x{rows}");
                            assert!(width >= 20 && l.view_rows >= DASHBOARD_MIN_ROWS);
                        }
                        None => assert_eq!(l.view_cols, cols, "{cols}x{rows}"),
                    }
                }
            }
        }
        // Tiny terminals give up the HUD before the view; nothing is drawn below the screen.
        assert_eq!(ViewerLayout::compute(80, 8, false).view_rows, 6);
        assert_eq!(ViewerLayout::compute(80, 5, false).hud_rows, 1);
        assert_eq!(ViewerLayout::compute(80, 2, false).hud_rows, 0);
        // A dashboard that does not fit is hidden, not squeezed.
        assert_eq!(ViewerLayout::compute(130, 8, true).dashboard, None);
        assert_eq!(ViewerLayout::compute(89, 40, true).dashboard, None);
        // Resizing recomputes everything, including the HUD position.
        let (big, small) = (
            ViewerLayout::compute(140, 40, true),
            ViewerLayout::compute(140, 20, true),
        );
        assert_eq!(big.view_rows, 38);
        assert_eq!(small.view_rows, 18);
        assert_eq!(
            small.hud_lines(STATUS, CONTROLS)[0].find("\x1b[19;1H"),
            Some(0)
        );
    }

    static RESTORED: AtomicUsize = AtomicUsize::new(0);
    fn count_restore() {
        RESTORED.fetch_add(1, Ordering::SeqCst);
    }

    /// A panic inside the viewer must restore the terminal before the message is printed, and
    /// the guard must survive being dropped during that very unwind (calling `set_hook` while
    /// panicking would abort the process).
    #[test]
    fn a_panic_restores_the_terminal_first() {
        let before = RESTORED.load(Ordering::SeqCst);
        let result = std::panic::catch_unwind(|| {
            let _guard = PanicHookGuard::install(count_restore);
            panic!("viewer blew up");
        });
        assert!(result.is_err());
        assert!(
            RESTORED.load(Ordering::SeqCst) > before,
            "the panic hook did not restore the terminal"
        );

        // Dropped normally, the guard reinstates the previous hook.
        drop(PanicHookGuard::install(count_restore));
        let after = RESTORED.load(Ordering::SeqCst);
        let _ = std::panic::catch_unwind(|| panic!("after the viewer"));
        assert_eq!(
            RESTORED.load(Ordering::SeqCst),
            after,
            "the restoring hook outlived the viewer"
        );
    }
}
