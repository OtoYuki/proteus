use crate::error::RenderError;
use crate::geometry::mesh::TriangleMesh;
use crate::rasterizer::buffer::{ColorRGB, Framebuffer};
use crate::rasterizer::camera::OrbitCamera;
use crate::rasterizer::pipeline::Rasterizer;
use crate::rasterizer::shader::ColorScheme;
use crate::terminal::halfblock::HalfBlockRenderer;
use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use std::io::{self, Write};
use std::time::{Duration, Instant};

struct RawTerminalGuard;

impl Drop for RawTerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, Show);
    }
}

pub struct ViewerConfig {
    pub title: String,
    pub initial_color_scheme: ColorScheme,
    pub auto_rotate: bool,
    pub secondary_mesh: Option<(TriangleMesh, ColorRGB)>,
    pub rmsd: Option<f64>,
}

impl Default for ViewerConfig {
    fn default() -> Self {
        Self {
            title: "Proteus 3D Viewer".to_string(),
            initial_color_scheme: ColorScheme::Plddt,
            auto_rotate: true,
            secondary_mesh: None,
            rmsd: None,
        }
    }
}

/// Run interactive 60 FPS 3D terminal protein ribbon viewer.
pub fn run_interactive_viewer(
    mesh: &TriangleMesh,
    mut camera: OrbitCamera,
    config: ViewerConfig,
) -> Result<(), RenderError> {
    enable_raw_mode().map_err(|e| RenderError::Terminal(e.to_string()))?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, Hide)
        .map_err(|e| RenderError::Terminal(e.to_string()))?;
    let _guard = RawTerminalGuard;

    let (mut term_cols, mut term_rows) =
        crossterm::terminal::size().map_err(|e| RenderError::Terminal(e.to_string()))?;

    // Reserve 2 rows at the bottom for HUD and controls
    let hud_height = 2u16;
    let render_rows = term_rows.saturating_sub(hud_height).max(10);
    // Halfblock resolution: 1 character row = 2 pixel rows
    let mut fb = Framebuffer::new(term_cols as usize, (render_rows * 2) as usize);

    let mut rasterizer = Rasterizer::new(config.initial_color_scheme);
    let mut compositor = HalfBlockRenderer::new();
    let mut auto_rotate = config.auto_rotate;
    let mut color_scheme = config.initial_color_scheme;

    let mut out_buf = String::with_capacity(64 * 1024);
    let mut last_frame = Instant::now();
    let mut fps = 0.0f32;
    let mut frame_count = 0usize;
    let mut fps_timer = Instant::now();

    loop {
        // Handle input events
        let timeout = Duration::from_millis(16);
        if event::poll(timeout).map_err(|e| RenderError::Terminal(e.to_string()))? {
            match event::read().map_err(|e| RenderError::Terminal(e.to_string()))? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                    KeyCode::Char(' ') => auto_rotate = !auto_rotate,
                    KeyCode::Char('c') => {
                        color_scheme = match color_scheme {
                            ColorScheme::Plddt => ColorScheme::SecondaryStructure,
                            ColorScheme::SecondaryStructure => ColorScheme::Rainbow,
                            ColorScheme::Rainbow => ColorScheme::Plddt,
                            ColorScheme::Solid(_) => ColorScheme::Plddt,
                        };
                        rasterizer.color_scheme = color_scheme;
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
                Event::Resize(new_cols, new_rows) => {
                    term_cols = new_cols;
                    term_rows = new_rows;
                    let new_render_rows = term_rows.saturating_sub(hud_height).max(10);
                    fb.resize(term_cols as usize, (new_render_rows * 2) as usize);
                    let _ = execute!(
                        stdout,
                        crossterm::terminal::Clear(crossterm::terminal::ClearType::All)
                    );
                }
                _ => {}
            }
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

        // Render frame
        fb.clear(ColorRGB::BLACK);
        rasterizer.render(mesh, &camera, &mut fb);

        // Render superimposed secondary mesh if present
        if let Some((ref sec_mesh, sec_color)) = config.secondary_mesh {
            rasterizer.render_with_scheme(
                sec_mesh,
                &camera,
                &mut fb,
                ColorScheme::Solid(sec_color),
            );
        }

        // Compose to terminal
        out_buf.clear();
        compositor.render_differential(&fb, &mut out_buf, 0, 0);

        // Draw HUD status lines
        let hud_scheme = match color_scheme {
            ColorScheme::Plddt => "pLDDT Confidence",
            ColorScheme::SecondaryStructure => "Secondary Structure",
            ColorScheme::Rainbow => "N->C Rainbow",
            ColorScheme::Solid(_) => "Solid",
        };

        let auto_status = if auto_rotate { "ON " } else { "OFF" };
        let total_triangles = mesh.triangle_count()
            + config
                .secondary_mesh
                .as_ref()
                .map_or(0, |(m, _)| m.triangle_count());

        let status_row1 = if let Some(rmsd) = config.rmsd {
            format!(
                " {} | Superimposed RMSD: {:.3} Å | Tris: {} | Spin: {} | {:.0} FPS",
                config.title, rmsd, total_triangles, auto_status, fps
            )
        } else {
            format!(
                " {} | Triangles: {} | Color: {} | Spin: {} | {:.0} FPS",
                config.title, total_triangles, hud_scheme, auto_status, fps
            )
        };
        let status_row2 = " [h/j/k/l/arrows] Orbit | [+/-] Zoom | [Space] Spin | [c] Color | [r] Reset | [q] Quit";

        let _ = execute!(
            stdout,
            MoveTo(0, term_rows.saturating_sub(2)),
            crossterm::style::SetForegroundColor(crossterm::style::Color::Cyan),
            crossterm::style::Print(format!(
                "{:<width$}",
                status_row1,
                width = term_cols as usize
            )),
            MoveTo(0, term_rows.saturating_sub(1)),
            crossterm::style::SetForegroundColor(crossterm::style::Color::DarkGrey),
            crossterm::style::Print(format!(
                "{:<width$}",
                status_row2,
                width = term_cols as usize
            )),
            crossterm::style::ResetColor
        );

        if !out_buf.is_empty() {
            let _ = stdout.write_all(out_buf.as_bytes());
            let _ = stdout.flush();
        }
    }

    Ok(())
}
