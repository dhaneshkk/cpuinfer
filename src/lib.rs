//! CPU-only YOLOX PPE detector: CLAHE + letterbox preprocessing, YOLOX head
//! decoding, class-wise NMS and MCC evaluation. The OpenVINO glue lives in
//! `main.rs`; everything here is pure Rust and unit-tested.
pub mod metrics;
pub mod postprocess;
pub mod preprocess;
pub mod types;
