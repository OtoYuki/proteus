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

    /// One frame of the interactive viewer, appended to `out`: `fb` as RGBA with the empty
    /// (pure black) pixels transparent, so the terminal's own background shows, zlib-compressed,
    /// shown at the cursor stretched over `cols` × `rows` cells. Sending it again with the same
    /// `image_id` replaces the picture in place. The cursor does not move (`C=1`), and replies
    /// are suppressed (`q=2`) so they never reach the viewer's key input.
    pub fn frame(fb: &Framebuffer, cols: u16, rows: u16, image_id: u32, out: &mut String) {
        use std::fmt::Write as _;
        let mut rgba = Vec::with_capacity(fb.width * fb.height * 4);
        for c in &fb.colors {
            let empty = c.r == 0 && c.g == 0 && c.b == 0;
            rgba.extend_from_slice(&[c.r, c.g, c.b, if empty { 0 } else { 255 }]);
        }
        let mut z = flate2::write::ZlibEncoder::new(
            Vec::with_capacity(rgba.len() / 4),
            flate2::Compression::fast(),
        );
        let _ = z.write_all(&rgba);
        let packed = z.finish().unwrap_or_default();
        let encoded = BASE64.encode(&packed);
        let bytes = encoded.as_bytes();
        let n = bytes.len().div_ceil(4096).max(1);
        for (k, chunk) in bytes.chunks(4096).enumerate() {
            let more = u8::from(k + 1 < n);
            let chunk = std::str::from_utf8(chunk).unwrap_or("");
            if k == 0 {
                let _ = write!(
                    out,
                    "\x1b_Ga=T,f=32,o=z,i={image_id},p=1,s={},v={},c={cols},r={rows},C=1,q=2,m={more};{chunk}\x1b\\",
                    fb.width, fb.height
                );
            } else {
                let _ = write!(out, "\x1b_Gm={more};{chunk}\x1b\\");
            }
        }
    }

    /// Remove the viewer's picture (on exit, or when falling back to text).
    pub fn delete(image_id: u32, out: &mut String) {
        out.push_str(&format!("\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\"));
    }
}

#[cfg(test)]
mod frame_tests {
    use super::*;
    use crate::rasterizer::buffer::ColorRGB;

    /// The frame decodes back to the pixels, with empty ones transparent.
    #[test]
    fn a_frame_round_trips_with_transparent_background() {
        use std::io::Read;
        let mut fb = Framebuffer::new(3, 2);
        fb.colors[1] = ColorRGB::new(10, 20, 30);
        let mut out = String::new();
        KittyRenderer::frame(&fb, 10, 4, 7, &mut out);
        assert!(out.starts_with("\x1b_Ga=T,f=32,o=z,i=7,p=1,s=3,v=2,c=10,r=4,C=1,q=2,m=0;"));
        let payload = &out[out.find(';').unwrap() + 1..out.rfind("\x1b\\").unwrap()];
        let packed = BASE64.decode(payload).unwrap();
        let mut rgba = Vec::new();
        flate2::read::ZlibDecoder::new(&packed[..])
            .read_to_end(&mut rgba)
            .unwrap();
        assert_eq!(rgba.len(), 3 * 2 * 4);
        assert_eq!(&rgba[0..4], &[0, 0, 0, 0]);
        assert_eq!(&rgba[4..8], &[10, 20, 30, 255]);
    }
}
