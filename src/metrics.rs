//! Matthews correlation coefficient for PPE-compliance evaluation.
//!
//! Box-level detection has no true negatives, so MCC is computed on the
//! image-level question "is class k present in this image?", which is exactly
//! the decision a PPE-compliance check makes (e.g. "person without helmet").
use crate::types::Detection;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Confusion {
    pub tp: u64,
    pub fp: u64,
    pub tn: u64,
    pub fn_: u64,
}

impl Confusion {
    pub fn update(&mut self, predicted: bool, actual: bool) {
        match (predicted, actual) {
            (true, true) => self.tp += 1,
            (true, false) => self.fp += 1,
            (false, false) => self.tn += 1,
            (false, true) => self.fn_ += 1,
        }
    }

    /// MCC in [-1, 1]. Returns 0 when any marginal is empty (undefined case,
    /// the scikit-learn convention).
    pub fn mcc(&self) -> f64 {
        let (tp, fp, tn, fn_) = (self.tp as f64, self.fp as f64, self.tn as f64, self.fn_ as f64);
        let denom = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
        if denom == 0.0 {
            0.0
        } else {
            ((tp * tn - fp * fn_) / denom).clamp(-1.0, 1.0)
        }
    }
}

/// Per-class image-level presence confusion over a dataset.
/// `predictions[i]` are the detections for image i, `ground_truth[i]` the
/// class ids annotated in image i.
pub fn presence_confusion(
    predictions: &[Vec<Detection>],
    ground_truth: &[Vec<usize>],
    num_classes: usize,
    score_threshold: f32,
) -> Vec<Confusion> {
    assert_eq!(predictions.len(), ground_truth.len(), "one prediction list per image");
    let mut out = vec![Confusion::default(); num_classes];
    for (preds, gts) in predictions.iter().zip(ground_truth) {
        for (k, conf) in out.iter_mut().enumerate() {
            let p = preds.iter().any(|d| d.class_id == k && d.score >= score_threshold);
            conf.update(p, gts.contains(&k));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcc_extremes_and_undefined() {
        let perfect = Confusion { tp: 5, fp: 0, tn: 5, fn_: 0 };
        let inverse = Confusion { tp: 0, fp: 5, tn: 0, fn_: 5 };
        assert_eq!(perfect.mcc(), 1.0);
        assert_eq!(inverse.mcc(), -1.0);
        assert_eq!(Confusion { tp: 3, fp: 0, tn: 0, fn_: 0 }.mcc(), 0.0);
        let c = Confusion { tp: 6, fp: 2, tn: 3, fn_: 1 };
        // (18 - 2) / sqrt(8 * 7 * 5 * 4)
        assert!((c.mcc() - 16.0 / 1120f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn presence_confusion_counts() {
        let det = |c, s| Detection { bbox: [0.0, 0.0, 1.0, 1.0], score: s, class_id: c };
        let preds = vec![vec![det(0, 0.9)], vec![det(1, 0.2)], vec![]];
        let gts = vec![vec![0], vec![1], vec![0]];
        let c = presence_confusion(&preds, &gts, 2, 0.3);
        assert_eq!(c[0], Confusion { tp: 1, fp: 0, tn: 1, fn_: 1 });
        assert_eq!(c[1], Confusion { tp: 0, fp: 0, tn: 2, fn_: 1 });
    }
}
