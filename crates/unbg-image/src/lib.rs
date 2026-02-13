#[derive(Debug, Clone, Copy)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

pub fn estimate_rgba_bytes(size: ImageSize) -> u64 {
    (size.width as u64)
        .saturating_mul(size.height as u64)
        .saturating_mul(4)
}

pub fn clamp_to_max_pixels(size: ImageSize, max_pixels: u32) -> ImageSize {
    let pixels = size.width.saturating_mul(size.height);
    if pixels <= max_pixels || size.width == 0 || size.height == 0 {
        return size;
    }

    let aspect = size.width as f32 / size.height as f32;
    let new_height = ((max_pixels as f32 / aspect).sqrt()).max(1.0) as u32;
    let new_width = ((new_height as f32) * aspect).max(1.0) as u32;
    ImageSize {
        width: new_width,
        height: new_height,
    }
}
