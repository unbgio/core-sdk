use std::io::Cursor;

use image::ImageReader;
use thiserror::Error;

pub const MAX_ENCODED_IMAGE_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_IMAGE_DIMENSION: u32 = 16_384;
pub const MAX_IMAGE_PIXELS: u64 = 40_000_000;

#[derive(Debug, Clone, Copy)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Error)]
pub enum ImageValidationError {
    #[error("encoded image is {actual} bytes; maximum is {maximum} bytes")]
    EncodedSize { actual: u64, maximum: u64 },
    #[error("image dimensions must be non-zero")]
    EmptyDimensions,
    #[error("image is {width}x{height}; maximum dimension is {maximum}")]
    Dimension {
        width: u32,
        height: u32,
        maximum: u32,
    },
    #[error("image has {actual} pixels; maximum is {maximum} pixels")]
    PixelCount { actual: u64, maximum: u64 },
    #[error("invalid image header: {0}")]
    Header(String),
}

pub fn validate_encoded_image_size(size: u64) -> Result<(), ImageValidationError> {
    if size > MAX_ENCODED_IMAGE_BYTES {
        return Err(ImageValidationError::EncodedSize {
            actual: size,
            maximum: MAX_ENCODED_IMAGE_BYTES,
        });
    }
    Ok(())
}

pub fn validate_image_size(size: ImageSize) -> Result<(), ImageValidationError> {
    if size.width == 0 || size.height == 0 {
        return Err(ImageValidationError::EmptyDimensions);
    }
    if size.width > MAX_IMAGE_DIMENSION || size.height > MAX_IMAGE_DIMENSION {
        return Err(ImageValidationError::Dimension {
            width: size.width,
            height: size.height,
            maximum: MAX_IMAGE_DIMENSION,
        });
    }
    let pixels = u64::from(size.width) * u64::from(size.height);
    if pixels > MAX_IMAGE_PIXELS {
        return Err(ImageValidationError::PixelCount {
            actual: pixels,
            maximum: MAX_IMAGE_PIXELS,
        });
    }
    Ok(())
}

pub fn inspect_encoded_image(bytes: &[u8]) -> Result<ImageSize, ImageValidationError> {
    validate_encoded_image_size(bytes.len() as u64)?;
    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| ImageValidationError::Header(error.to_string()))?;
    let (width, height) = reader
        .into_dimensions()
        .map_err(|error| ImageValidationError::Header(error.to_string()))?;
    let size = ImageSize { width, height };
    validate_image_size(size)?;
    Ok(size)
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageFormat, RgbImage};

    #[test]
    fn inspects_valid_image_headers() {
        let image = DynamicImage::ImageRgb8(RgbImage::new(3, 2));
        let mut encoded = Vec::new();
        image
            .write_to(&mut Cursor::new(&mut encoded), ImageFormat::Png)
            .expect("test image should encode");

        let size = inspect_encoded_image(&encoded).expect("header should be valid");
        assert_eq!((size.width, size.height), (3, 2));
    }

    #[test]
    fn rejects_oversized_envelopes_and_dimensions() {
        assert!(validate_encoded_image_size(MAX_ENCODED_IMAGE_BYTES + 1).is_err());
        assert!(validate_image_size(ImageSize {
            width: 8_001,
            height: 5_000,
        })
        .is_err());
    }
}
