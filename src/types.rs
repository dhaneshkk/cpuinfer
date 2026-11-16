/// Letterbox transform applied in preprocessing.
///
/// `pad_x` / `pad_y` are the *integer* pixel offsets actually used when the
/// resized image was pasted onto the canvas (stored as f32 for arithmetic),
/// so `orig = (model - pad) / scale` is exact with respect to the canvas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LetterboxParams {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub orig_width: u32,
    pub orig_height: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// [x1, y1, x2, y2] in original-image pixel coordinates, clipped to the image.
    pub bbox: [f32; 4],
    pub score: f32,
    pub class_id: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Raw YOLOX head: [tx, ty, tw, th, obj_logit, cls_logits..]; needs grid
    /// offset, exp, stride and sigmoid.
    RawLogits,
    /// Exported with decode_in_inference: [cx, cy, w, h, obj, cls..] in
    /// model-input pixels with sigmoid already applied.
    Decoded,
}
