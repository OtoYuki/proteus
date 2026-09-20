use crate::rasterizer::buffer::Framebuffer;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use std::io::{self, Write};

/// Kitty Graphics Protocol APC encoder.
/// Directly streams 24-bit raw RGB pixel payloads to modern terminal emulators
/// (Kitty, WezTerm, Ghostty) in standard 4096-byte Base64 chunks.
pub struct KittyRenderer;

impl KittyRenderer {
    /// Detect if the running terminal emulator supports Kitty graphics protocol.
    pub fn is_supported() -> bool {
        if let Ok(term) = std::env::var("TERM") {
            if term.contains("kitty") || term.contains("ghostty") || term.contains("wezterm") {
                return true;
            }
        }
        if std::env::var("KITTY_PID").is_ok() || std::env::var("KITTY_WINDOW_ID").is_ok() {
            return true;
        }
        false
    }

    /// Render framebuffer directly to writer using Kitty graphics protocol.
    pub fn render(fb: &Framebuffer, out: &mut impl Write) -> io::Result<()> {
        let width = fb.width;
        let height = fb.height;

        // Extract raw 24-bit RGB bytes
        let mut raw_rgb = Vec::with_capacity(width * height * 3);
        for color in &fb.colors {
            raw_rgb.push(color.r);
            raw_rgb.push(color.g);
            raw_rgb.push(color.b);
        }

        let encoded = BASE64.encode(&raw_rgb);
        let bytes = encoded.as_bytes();
        let chunk_size = 4096;
        let mut offset = 0;
        let mut is_first = true;

        while offset < bytes.len() {
            let end = (offset + chunk_size).min(bytes.len());
            let is_last = end == bytes.len();
            let chunk = &bytes[offset..end];
            let chunk_str = std::str::from_utf8(chunk).unwrap_or("");

            if is_first {
                let m = if is_last { 0 } else { 1 };
                write!(
                    out,
                    "\x1b_Ga=T,f=24,s={},v={},m={};{}\x1b\\",
                    width, height, m, chunk_str
                )?;
                is_first = false;
            } else {
                let m = if is_last { 0 } else { 1 };
                write!(out, "\x1b_Gm={};{}\x1b\\", m, chunk_str)?;
            }

            offset = end;
        }

        out.flush()
    }
}
