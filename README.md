# yolox-ppe-cpu

CPU-only YOLOX PPE (personal protective equipment) detector in Rust on OpenVINO.

Pipeline: grayscale frame → CLAHE (OpenCV semantics, 8×8 tiles, clip 2.0) →
centred letterbox (pad 114) → 3-channel CHW f32 → OpenVINO CPU → YOLOX head
decode → class-wise NMS → boxes in original-image pixels (JSON on stdout).

```
cargo build --release
./target/release/yolox-ppe-cpu --model ir_model/yolox_ppe.xml --image frame.png \
    --size 416 --classes 3 --format raw
```

`--format raw` is for a plain YOLOX export; use `--format decoded` if the ONNX
was exported with `--decode_in_inference`. A probe warns if the output looks
like the other format. The `openvino` crate loads `libopenvino_c` at runtime,
so the OpenVINO runtime must be installed (or the pip wheel's `libs/` directory
must be on `LD_LIBRARY_PATH`).

## Layout

| Path | Contents |
|---|---|
| `src/preprocess.rs` | CLAHE, letterbox geometry, CHW tensor |
| `src/postprocess.rs` | Output-format probe, YOLOX decode, NMS, IoU |
| `src/metrics.rs` | Image-level per-class presence MCC (Matthews correlation) |
| `src/main.rs` | CLI + OpenVINO inference |
| `lean/YoloxPpe.lean` | Index-safety lemmas (letterbox fit, canvas bounds, CLAHE tile bounds, confusion totals) |
| `notebooks/yolox_ppe_cpu_colab.ipynb` | Single Colab cell: build, convert ONNX→IR, serve `/v1/chat/completions` |

## Evaluation

Box-level detection has no true negatives, so MCC is computed per class on the
image-level question "is class k present?" (`metrics::presence_confusion`),
which is the decision a compliance check actually makes.

## OpenAI-compatible endpoint

The notebook serves `POST /v1/chat/completions`. Send the frame as an
`image_url` content part (data URL or http URL); the assistant message content
is `{"detections":[{"class_id","label","score","bbox":[x1,y1,x2,y2]}]}`.
Optional extra request fields: `score_threshold`, `iou_threshold`.
