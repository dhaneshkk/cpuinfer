use crate::types::LetterboxParams;
use anyhow::{bail, Context, Result};
use image::{imageops::FilterType, GrayImage};

/// Grey value YOLOX uses for letterbox padding.
pub const PAD_VALUE: u8 = 114;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Contrast-limited adaptive histogram equalisation, following OpenCV's
/// `cv::createCLAHE` semantics (clip limit relative to the mean bin height,
/// uniform redistribution of clipped mass, LUT = cdf * 255 / tile_area,
/// bilinear interpolation between the four nearest tile centres).
///
/// Unlike OpenCV, a non-divisible image is not border-padded; the last tile
/// row/column is simply smaller.
pub fn apply_clahe(img: &GrayImage, tile_grid_size: (usize, usize), clip_limit: f32) -> Result<GrayImage> {
    let (width, height) = img.dimensions();
    let (w, h) = (width as usize, height as usize);
    let (req_tx, req_ty) = tile_grid_size;
    if req_tx == 0 || req_ty == 0 {
        bail!("CLAHE tile grid must be non-zero, got {req_tx}x{req_ty}");
    }
    if !clip_limit.is_finite() || clip_limit < 0.0 {
        bail!("CLAHE clip limit must be finite and >= 0, got {clip_limit}");
    }
    if w == 0 || h == 0 {
        return Ok(img.clone());
    }

    // Never ask for more tiles than pixels, then recompute the tile count so
    // that every tile is non-empty (see lean/YoloxPpe.lean: `tile_index_lt`).
    let tile_w = w.div_ceil(req_tx.min(w));
    let tile_h = h.div_ceil(req_ty.min(h));
    let tiles_x = w.div_ceil(tile_w);
    let tiles_y = h.div_ceil(tile_h);

    let src = img.as_raw();
    let mut luts = vec![[0u8; 256]; tiles_x * tiles_y];

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let (x0, y0) = (tx * tile_w, ty * tile_h);
            let (x1, y1) = ((x0 + tile_w).min(w), (y0 + tile_h).min(h));
            let area = ((x1 - x0) * (y1 - y0)) as f32;

            let mut hist = [0f32; 256];
            for y in y0..y1 {
                for &v in &src[y * w + x0..y * w + x1] {
                    hist[v as usize] += 1.0;
                }
            }

            if clip_limit > 0.0 {
                let clip = (clip_limit * area / 256.0).max(1.0);
                let mut excess = 0.0;
                for bin in hist.iter_mut() {
                    if *bin > clip {
                        excess += *bin - clip;
                        *bin = clip;
                    }
                }
                let add = excess / 256.0;
                for bin in hist.iter_mut() {
                    *bin += add;
                }
            }

            let lut = &mut luts[ty * tiles_x + tx];
            let k = 255.0 / area;
            let mut cdf = 0.0;
            for (i, &bin) in hist.iter().enumerate() {
                cdf += bin;
                lut[i] = (cdf * k).round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    let mut out = vec![0u8; w * h];
    let (inv_tw, inv_th) = (1.0 / tile_w as f32, 1.0 / tile_h as f32);
    for y in 0..h {
        let tyf = y as f32 * inv_th - 0.5;
        let ty1f = tyf.floor();
        let ya = tyf - ty1f;
        let ty1 = ((ty1f as isize).max(0) as usize).min(tiles_y - 1);
        let ty2 = ((ty1f as isize + 1) as usize).min(tiles_y - 1);
        for x in 0..w {
            let txf = x as f32 * inv_tw - 0.5;
            let tx1f = txf.floor();
            let xa = txf - tx1f;
            let tx1 = ((tx1f as isize).max(0) as usize).min(tiles_x - 1);
            let tx2 = ((tx1f as isize + 1) as usize).min(tiles_x - 1);

            let v = src[y * w + x] as usize;
            let l = |ty: usize, tx: usize| luts[ty * tiles_x + tx][v] as f32;
            let top = l(ty1, tx1) * (1.0 - xa) + l(ty1, tx2) * xa;
            let bot = l(ty2, tx1) * (1.0 - xa) + l(ty2, tx2) * xa;
            out[y * w + x] = (top * (1.0 - ya) + bot * ya).round().clamp(0.0, 255.0) as u8;
        }
    }

    GrayImage::from_raw(width, height, out).context("CLAHE output buffer has wrong length")
}

/// Pure letterbox geometry: scaled size and centring offsets.
/// Guarantees `new_w + pad_x <= target_w` and `new_h + pad_y <= target_h`
/// (proved in lean/YoloxPpe.lean: `letterbox_fits`).
pub fn letterbox_geometry(orig: (u32, u32), target: (u32, u32)) -> (f32, u32, u32, u32, u32) {
    let (ow, oh) = orig;
    let (tw, th) = target;
    let scale = (tw as f32 / ow as f32).min(th as f32 / oh as f32);
    let new_w = ((ow as f32 * scale).round() as u32).clamp(1, tw);
    let new_h = ((oh as f32 * scale).round() as u32).clamp(1, th);
    (scale, new_w, new_h, (tw - new_w) / 2, (th - new_h) / 2)
}

/// Grayscale frame -> CLAHE -> centred letterbox -> 3-channel CHW f32 tensor.
///
/// All three channels carry the same luma value (the model was trained on
/// 3-channel input). With `normalize`, ImageNet mean/std is applied per
/// channel; otherwise values stay in [0, 255] (YOLOX default, legacy=False).
pub fn preprocess_monochrome_frame(
    gray_data: &[u8],
    orig_width: u32,
    orig_height: u32,
    target_size: (u32, u32),
    normalize: bool,
) -> Result<(Vec<f32>, LetterboxParams)> {
    let (target_w, target_h) = target_size;
    if orig_width == 0 || orig_height == 0 || target_w == 0 || target_h == 0 {
        bail!("Frame and target sizes must be non-zero: frame {orig_width}x{orig_height}, target {target_w}x{target_h}");
    }
    let expected_len = (orig_width as usize)
        .checked_mul(orig_height as usize)
        .context("Frame dimensions overflow usize")?;
    if gray_data.len() != expected_len {
        bail!(
            "Malformed frame buffer: got {} bytes, expected {expected_len} ({orig_width}x{orig_height})",
            gray_data.len()
        );
    }

    let gray = GrayImage::from_raw(orig_width, orig_height, gray_data.to_vec())
        .context("Failed to wrap grayscale frame")?;
    let enhanced = apply_clahe(&gray, (8, 8), 2.0)?;

    let (scale, new_w, new_h, pad_x, pad_y) = letterbox_geometry((orig_width, orig_height), target_size);
    let resized = image::imageops::resize(&enhanced, new_w, new_h, FilterType::Triangle);

    let (tw, nw) = (target_w as usize, new_w as usize);
    let mut canvas = vec![PAD_VALUE; tw * target_h as usize];
    for (row, src_row) in resized.as_raw().chunks_exact(nw).enumerate() {
        let start = (row + pad_y as usize) * tw + pad_x as usize;
        canvas[start..start + nw].copy_from_slice(src_row);
    }

    let plane = canvas.len();
    let mut chw = vec![0.0f32; 3 * plane];
    for c in 0..3 {
        let dst = &mut chw[c * plane..(c + 1) * plane];
        if normalize {
            let (m, s) = (IMAGENET_MEAN[c], IMAGENET_STD[c]);
            for (d, &v) in dst.iter_mut().zip(&canvas) {
                *d = (v as f32 / 255.0 - m) / s;
            }
        } else {
            for (d, &v) in dst.iter_mut().zip(&canvas) {
                *d = v as f32;
            }
        }
    }

    Ok((
        chw,
        LetterboxParams {
            scale,
            pad_x: pad_x as f32,
            pad_y: pad_y as f32,
            orig_width,
            orig_height,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(w: u32, h: u32) -> GrayImage {
        GrayImage::from_raw(w, h, (0..w * h).map(|i| (i % 97) as u8).collect()).unwrap()
    }

    #[test]
    fn clahe_rejects_zero_grid() {
        assert!(apply_clahe(&ramp(16, 16), (0, 8), 2.0).is_err());
    }

    #[test]
    fn clahe_handles_degenerate_grids_without_panicking() {
        for &(w, h) in &[(1, 1), (3, 50), (50, 3), (10, 10), (17, 9), (1280, 720)] {
            for &g in &[(1, 1), (1, 8), (8, 1), (8, 8), (64, 64)] {
                let out = apply_clahe(&ramp(w, h), g, 2.0).unwrap();
                assert_eq!(out.dimensions(), (w, h));
            }
        }
    }

    #[test]
    fn clahe_constant_image_stays_constant() {
        let img = GrayImage::from_raw(64, 32, vec![128; 64 * 32]).unwrap();
        let out = apply_clahe(&img, (8, 8), 2.0).unwrap();
        let first = out.as_raw()[0];
        assert!(out.as_raw().iter().all(|&v| v == first));
    }

    #[test]
    fn letterbox_geometry_fits_and_centres() {
        let (s, nw, nh, px, py) = letterbox_geometry((1280, 720), (416, 416));
        assert!((s - 0.325).abs() < 1e-6);
        assert_eq!((nw, nh, px, py), (416, 234, 0, 91));
        for &(ow, oh) in &[(1, 1), (1, 10000), (10000, 1), (415, 417), (3, 7)] {
            let (_, nw, nh, px, py) = letterbox_geometry((ow, oh), (416, 416));
            assert!(nw + px <= 416 && nh + py <= 416 && nw >= 1 && nh >= 1);
        }
    }

    #[test]
    fn preprocess_shapes_padding_and_errors() {
        let frame = vec![128u8; 1280 * 720];
        let (t, lb) = preprocess_monochrome_frame(&frame, 1280, 720, (416, 416), false).unwrap();
        assert_eq!(t.len(), 3 * 416 * 416);
        assert_eq!((lb.pad_x, lb.pad_y), (0.0, 91.0));
        assert_eq!(t[0], PAD_VALUE as f32); // top padding band
        assert_eq!(t[416 * 200], t[2 * 416 * 416 + 416 * 200]); // channels equal

        assert!(preprocess_monochrome_frame(&frame[1..], 1280, 720, (416, 416), false).is_err());
        assert!(preprocess_monochrome_frame(&[], 0, 720, (416, 416), false).is_err());
    }
}
