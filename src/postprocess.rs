use crate::types::{Detection, LetterboxParams, OutputFormat};
use anyhow::{bail, Result};

pub const STRIDES: [usize; 3] = [8, 16, 32];

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Number of anchor points YOLOX emits for an (h, w) input.
pub fn n_anchors_for((h, w): (usize, usize)) -> usize {
    STRIDES.iter().map(|&s| (h / s) * (w / s)).sum()
}

fn check_input_size((h, w): (usize, usize)) -> Result<()> {
    if h == 0 || w == 0 || h % 32 != 0 || w % 32 != 0 {
        bail!("YOLOX input size must be a non-zero multiple of 32, got {h}x{w}");
    }
    Ok(())
}

/// (grid_x, grid_y, stride) for each anchor, in the order YOLOX concatenates
/// them: stride 8, 16, 32, each row-major over (y, x).
fn grids_and_strides((h, w): (usize, usize)) -> Vec<(usize, usize, usize)> {
    let mut grids = Vec::with_capacity(n_anchors_for((h, w)));
    for &s in &STRIDES {
        for gy in 0..h / s {
            for gx in 0..w / s {
                grids.push((gx, gy, s));
            }
        }
    }
    grids
}

fn check_len(raw: &[f32], num_classes: usize, input_size: (usize, usize)) -> Result<usize> {
    check_input_size(input_size)?;
    if num_classes == 0 {
        bail!("num_classes must be >= 1");
    }
    let channels = 5 + num_classes;
    let expected = n_anchors_for(input_size) * channels;
    if raw.len() != expected {
        bail!(
            "Output tensor has {} floats, expected {} anchors x {channels} channels = {expected}. \
             Wrong num_classes / input size, or the export has NMS baked into the graph.",
            raw.len(),
            n_anchors_for(input_size)
        );
    }
    Ok(channels)
}

/// Heuristic probe: raw objectness logits almost always leave [0, 1]
/// (they are strongly negative on background), sigmoid outputs never do.
/// Samples ~512 anchors spread evenly over all three strides.
pub fn detect_output_format(raw: &[f32], num_classes: usize, input_size: (usize, usize)) -> Result<OutputFormat> {
    let channels = check_len(raw, num_classes, input_size)?;
    let anchors = n_anchors_for(input_size);
    let step = (anchors / 512).max(1);
    let out_of_unit = (0..anchors)
        .step_by(step)
        .map(|a| raw[a * channels + 4])
        .any(|v| !(-0.01..=1.01).contains(&v));
    Ok(if out_of_unit { OutputFormat::RawLogits } else { OutputFormat::Decoded })
}

/// Decode the YOLOX head, map boxes back to original-image coordinates,
/// clip them to the image and run class-wise NMS.
pub fn decode_outputs(
    raw: &[f32],
    num_classes: usize,
    score_threshold: f32,
    iou_threshold: f32,
    letterbox: LetterboxParams,
    input_size: (usize, usize),
    format: OutputFormat,
) -> Result<Vec<Detection>> {
    let channels = check_len(raw, num_classes, input_size)?;
    if !(letterbox.scale.is_finite() && letterbox.scale > 0.0) {
        bail!("Letterbox scale must be finite and > 0, got {}", letterbox.scale);
    }
    let (max_x, max_y) = (letterbox.orig_width as f32, letterbox.orig_height as f32);

    let mut proposals = Vec::new();
    for (row, &(gx, gy, stride)) in raw.chunks_exact(channels).zip(&grids_and_strides(input_size)) {
        // argmax over logits == argmax over sigmoid (monotone), so take the
        // sigmoid once afterwards.
        let (best, cls_val) = row[5..]
            .iter()
            .copied()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |acc, (i, v)| if v > acc.1 { (i, v) } else { acc });

        let s = stride as f32;
        let (score, cx, cy, bw, bh) = match format {
            OutputFormat::RawLogits => (
                sigmoid(row[4]) * sigmoid(cls_val),
                (row[0] + gx as f32) * s,
                (row[1] + gy as f32) * s,
                row[2].exp() * s,
                row[3].exp() * s,
            ),
            OutputFormat::Decoded => (row[4] * cls_val, row[0], row[1], row[2], row[3]),
        };
        if score.is_nan() || score < score_threshold {
            continue;
        }

        let unmap_x = |v: f32| ((v - letterbox.pad_x) / letterbox.scale).clamp(0.0, max_x);
        let unmap_y = |v: f32| ((v - letterbox.pad_y) / letterbox.scale).clamp(0.0, max_y);
        let bbox = [
            unmap_x(cx - bw / 2.0),
            unmap_y(cy - bh / 2.0),
            unmap_x(cx + bw / 2.0),
            unmap_y(cy + bh / 2.0),
        ];
        if bbox.iter().any(|v| !v.is_finite()) || bbox[2] <= bbox[0] || bbox[3] <= bbox[1] {
            continue; // degenerate or entirely outside the original image
        }
        proposals.push(Detection { bbox, score, class_id: best });
    }

    Ok(nms(proposals, iou_threshold))
}

/// Greedy class-wise NMS; output sorted by descending score.
pub fn nms(mut dets: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut suppressed = vec![false; dets.len()];
    let mut kept = Vec::new();
    for i in 0..dets.len() {
        if suppressed[i] {
            continue;
        }
        for j in i + 1..dets.len() {
            if !suppressed[j]
                && dets[i].class_id == dets[j].class_id
                && calculate_iou(&dets[i].bbox, &dets[j].bbox) > iou_threshold
            {
                suppressed[j] = true;
            }
        }
        kept.push(dets[i].clone());
    }
    kept
}

/// IoU of two [x1, y1, x2, y2] boxes; always in [0, 1] (degenerate -> 0).
pub fn calculate_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union > 0.0 {
        (inter / union).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: (usize, usize) = (416, 416);
    const NC: usize = 3;
    const CH: usize = 5 + NC;

    fn identity_lb() -> LetterboxParams {
        LetterboxParams { scale: 1.0, pad_x: 0.0, pad_y: 0.0, orig_width: 416, orig_height: 416 }
    }

    fn raw_background() -> Vec<f32> {
        let mut raw = vec![0.0; n_anchors_for(SIZE) * CH];
        for row in raw.chunks_exact_mut(CH) {
            row[4] = -8.0; // background objectness logit
            row[5..].fill(-8.0);
        }
        raw
    }

    #[test]
    fn anchor_count_416() {
        assert_eq!(n_anchors_for(SIZE), 52 * 52 + 26 * 26 + 13 * 13);
        assert_eq!(grids_and_strides(SIZE).len(), 3549);
    }

    #[test]
    fn rejects_bad_lengths_and_sizes() {
        assert!(detect_output_format(&[0.0; 10], NC, SIZE).is_err());
        assert!(detect_output_format(&[], NC, (16, 16)).is_err());
        assert!(decode_outputs(&[], 0, 0.3, 0.45, identity_lb(), SIZE, OutputFormat::RawLogits).is_err());
    }

    #[test]
    fn probe_distinguishes_formats() {
        let raw = raw_background();
        assert_eq!(detect_output_format(&raw, NC, SIZE).unwrap(), OutputFormat::RawLogits);
        let decoded: Vec<f32> = raw.iter().map(|&v| if v < 0.0 { 0.001 } else { v }).collect();
        assert_eq!(detect_output_format(&decoded, NC, SIZE).unwrap(), OutputFormat::Decoded);
    }

    #[test]
    fn raw_decode_grid_offset_and_letterbox_unmap() {
        let mut raw = raw_background();
        // anchor (gx=10, gy=5) at stride 8 -> index 5*52+10
        let row = &mut raw[(5 * 52 + 10) * CH..][..CH];
        row[..4].copy_from_slice(&[0.5, 0.5, 2f32.ln(), 2f32.ln()]); // 16x16 box at (84, 44)
        row[4] = 8.0;
        row[5 + 2] = 8.0;
        let lb = LetterboxParams { scale: 0.5, pad_x: 4.0, pad_y: 2.0, orig_width: 1000, orig_height: 1000 };
        let dets = decode_outputs(&raw, NC, 0.3, 0.45, lb, SIZE, OutputFormat::RawLogits).unwrap();
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class_id, 2);
        let expect = [(76.0 - 4.0) / 0.5, (36.0 - 2.0) / 0.5, (92.0 - 4.0) / 0.5, (52.0 - 2.0) / 0.5];
        for (a, b) in dets[0].bbox.iter().zip(expect) {
            assert!((a - b).abs() < 1e-3, "{:?} vs {:?}", dets[0].bbox, expect);
        }
    }

    #[test]
    fn decoded_format_is_center_xywh_and_clipped() {
        let mut raw = vec![0.0; n_anchors_for(SIZE) * CH];
        raw[..CH].copy_from_slice(&[100.0, 100.0, 40.0, 20.0, 0.9, 0.1, 0.8, 0.2]);
        raw[CH..2 * CH].copy_from_slice(&[5.0, 5.0, 40.0, 40.0, 0.9, 0.9, 0.0, 0.0]);
        let mut dets = decode_outputs(&raw, NC, 0.3, 0.45, identity_lb(), SIZE, OutputFormat::Decoded).unwrap();
        dets.sort_by_key(|d| d.class_id);
        assert_eq!(dets.len(), 2);
        assert_eq!(dets[0].bbox, [0.0, 0.0, 25.0, 25.0]); // clipped at the image edge
        assert_eq!(dets[1].bbox, [80.0, 90.0, 120.0, 110.0]);
        assert!((dets[1].score - 0.72).abs() < 1e-6);
    }

    #[test]
    fn nms_is_class_wise_and_sorted() {
        let d = |x: f32, s: f32, c: usize| Detection { bbox: [x, 0.0, x + 10.0, 10.0], score: s, class_id: c };
        let kept = nms(vec![d(0.0, 0.5, 0), d(1.0, 0.9, 0), d(1.0, 0.7, 1), d(50.0, f32::NAN, 0)], 0.45);
        let scores: Vec<_> = kept.iter().filter(|k| !k.score.is_nan()).map(|k| k.score).collect();
        assert_eq!(scores, vec![0.9, 0.7]);
    }

    #[test]
    fn iou_bounds() {
        let a = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(calculate_iou(&a, &a), 1.0);
        assert_eq!(calculate_iou(&a, &[20.0, 20.0, 30.0, 30.0]), 0.0);
        assert_eq!(calculate_iou(&[0.0; 4], &[0.0; 4]), 0.0);
        assert!((calculate_iou(&a, &[5.0, 0.0, 15.0, 10.0]) - 1.0 / 3.0).abs() < 1e-6);
    }
}
