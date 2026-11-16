use anyhow::{bail, Context, Result};
use openvino::{Core, DeviceType, ElementType, Shape, Tensor};
use yolox_ppe_cpu::postprocess::{decode_outputs, detect_output_format};
use yolox_ppe_cpu::preprocess::preprocess_monochrome_frame;
use yolox_ppe_cpu::types::{Detection, OutputFormat};

const USAGE: &str = "usage: yolox-ppe-cpu [--model ir_model/yolox_ppe.xml] [--image FILE] \
[--size 416] [--classes 3] [--score 0.3] [--iou 0.45] [--format raw|decoded] [--normalize]
Without --image a synthetic 1280x720 grey frame is used. Detections are printed as JSON on stdout.";

struct Args {
    model: String,
    image: Option<String>,
    size: u32,
    num_classes: usize,
    score: f32,
    iou: f32,
    format: OutputFormat,
    normalize: bool,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        model: "./ir_model/yolox_ppe.xml".into(),
        image: None,
        size: 416,
        num_classes: 3,
        score: 0.3,
        iou: 0.45,
        format: OutputFormat::RawLogits,
        normalize: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().with_context(|| format!("{flag} needs a value\n{USAGE}"));
        match flag.as_str() {
            "--model" => a.model = val()?,
            "--image" => a.image = Some(val()?),
            "--size" => a.size = val()?.parse().context("--size")?,
            "--classes" => a.num_classes = val()?.parse().context("--classes")?,
            "--score" => a.score = val()?.parse().context("--score")?,
            "--iou" => a.iou = val()?.parse().context("--iou")?,
            "--format" => {
                a.format = match val()?.as_str() {
                    "raw" => OutputFormat::RawLogits,
                    "decoded" => OutputFormat::Decoded,
                    other => bail!("--format must be raw|decoded, got {other}"),
                }
            }
            "--normalize" => a.normalize = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => bail!("unknown argument {other}\n{USAGE}"),
        }
    }
    Ok(a)
}

/// i64 tensor dims -> usize, rejecting negative (dynamic) axes instead of wrapping.
fn dims_usize(shape: &Shape) -> Result<Vec<usize>> {
    shape
        .get_dimensions()
        .iter()
        .map(|&d| usize::try_from(d))
        .collect::<Result<Vec<_>, _>>()
        .context("Tensor shape has a negative (dynamic) dimension; re-export with a fixed input size")
}

fn to_json(dets: &[Detection]) -> String {
    let items: Vec<String> = dets
        .iter()
        .map(|d| {
            format!(
                "{{\"class_id\":{},\"score\":{:.4},\"bbox\":[{:.1},{:.1},{:.1},{:.1}]}}",
                d.class_id, d.score, d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]
            )
        })
        .collect();
    format!("{{\"detections\":[{}]}}", items.join(","))
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let size = args.size as usize;

    let (gray, w, h) = match &args.image {
        Some(path) => {
            let img = image::open(path).with_context(|| format!("Failed to read image {path}"))?.to_luma8();
            let (w, h) = img.dimensions();
            (img.into_raw(), w, h)
        }
        None => (vec![128u8; 1280 * 720], 1280, 720),
    };

    eprintln!("Preprocessing {w}x{h} frame...");
    let (input, letterbox) = preprocess_monochrome_frame(&gray, w, h, (args.size, args.size), args.normalize)?;

    eprintln!("Loading OpenVINO model {}...", args.model);
    let weights = args
        .model
        .strip_suffix(".xml")
        .map(|stem| format!("{stem}.bin"))
        .context("--model must point at an OpenVINO IR .xml file")?;
    let mut core = Core::new().context("Failed to initialise OpenVINO (is the runtime library installed / on the loader path?)")?;
    let model = core
        .read_model_from_file(&args.model, &weights)
        .with_context(|| format!("Failed to load IR model {} / {weights}", args.model))?;
    let input_name = model.get_input_by_index(0)?.get_name()?;
    let output_name = model.get_output_by_index(0)?.get_name()?;

    let mut compiled = core.compile_model(&model, DeviceType::CPU).context("Failed to compile model for CPU")?;
    let mut request = compiled.create_infer_request()?;

    let in_shape = Shape::new(&[1, 3, size as i64, size as i64])?;
    let mut tensor = Tensor::new(ElementType::F32, &in_shape)?;
    tensor.get_data_mut::<f32>()?.copy_from_slice(&input);
    request.set_tensor(&input_name, &tensor)?;

    eprintln!("Running inference...");
    request.infer().context("Inference failed (does the model expect a 1x3xSxS f32 input?)")?;

    let output = request.get_tensor(&output_name)?;
    let out_dims = dims_usize(&output.get_shape()?)?;
    let channels = 5 + args.num_classes;
    match out_dims.as_slice() {
        [1, _, c] | [_, c] if *c == channels => {}
        _ => bail!("Unexpected output shape {out_dims:?}; expected [1, anchors, {channels}]"),
    }
    let raw = output.get_data::<f32>()?;

    let probed = detect_output_format(raw, args.num_classes, (size, size))?;
    if probed != args.format {
        eprintln!(
            "[WARN] --format is {:?} but the output looks like {probed:?}. \
             Check the ONNX export flag --decode_in_inference.",
            args.format
        );
    }

    let dets = decode_outputs(raw, args.num_classes, args.score, args.iou, letterbox, (size, size), args.format)?;
    eprintln!("Detected {} PPE objects", dets.len());
    println!("{}", to_json(&dets));
    Ok(())
}
