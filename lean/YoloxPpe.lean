/-
  Index-safety lemmas for yolox-ppe-cpu, core Lean 4 only (no Mathlib).
  NOT YET MACHINE-CHECKED: no Lean toolchain was available when this was
  written. Check with:  lean lean/YoloxPpe.lean
  Each lemma names the Rust code that relies on it.
-/

/-- preprocess.rs `letterbox_geometry`: with `pad = (t - n) / 2` and
    `n ≤ t` (guaranteed by `.clamp(1, t)`), the pasted image fits. -/
theorem letterbox_fits (t n : Nat) (h : n ≤ t) : n + (t - n) / 2 ≤ t := by
  omega

/-- preprocess.rs canvas paste: every write index
    `(row + pad_y) * tw + pad_x + col` is inside the `tw * th` canvas. -/
theorem canvas_index_lt (tw th nw nh px py row col : Nat)
    (hw : nw + px ≤ tw) (hh : nh + py ≤ th) (hr : row < nh) (hc : col < nw) :
    (row + py) * tw + (px + col) < tw * th := by
  have h2 : row + py + 1 ≤ th := by omega
  have h3 : (row + py + 1) * tw ≤ th * tw := Nat.mul_le_mul_right tw h2
  have h4 : (row + py + 1) * tw = (row + py) * tw + tw := by
    rw [Nat.add_mul, Nat.one_mul]
  rw [Nat.mul_comm tw th]
  omega

/-- `usize::div_ceil` covers the range: `w ≤ ceil(w / k) * k`. -/
theorem div_ceil_cover (w k : Nat) (hk : 0 < k) :
    w ≤ ((w + k - 1) / k) * k := by
  have h1 := Nat.div_add_mod (w + k - 1) k
  have h2 := Nat.mod_lt (w + k - 1) hk
  rw [Nat.mul_comm]
  omega

/-- preprocess.rs `apply_clahe`: with `tiles = w.div_ceil(tile_w)`, every
    pixel column `x < w` lands in a tile `x / tile_w < tiles`, so the LUT
    index `ty * tiles_x + tx` is in bounds. -/
theorem tile_index_lt (x w k : Nat) (hk : 0 < k) (hx : x < w) :
    x / k < (w + k - 1) / k :=
  (Nat.div_lt_iff_lt_mul hk).mpr (Nat.lt_of_lt_of_le hx (div_ceil_cover w k hk))

/-- metrics.rs: the per-image confusion update increments exactly one of
    TP/FP/TN/FN, so the four counts always sum to the number of images. -/
def confusionStep (c : Nat × Nat × Nat × Nat) : Bool → Bool → Nat × Nat × Nat × Nat
  | true,  true  => (c.1 + 1, c.2.1, c.2.2.1, c.2.2.2)
  | true,  false => (c.1, c.2.1 + 1, c.2.2.1, c.2.2.2)
  | false, false => (c.1, c.2.1, c.2.2.1 + 1, c.2.2.2)
  | false, true  => (c.1, c.2.1, c.2.2.1, c.2.2.2 + 1)

def total (c : Nat × Nat × Nat × Nat) : Nat := c.1 + c.2.1 + c.2.2.1 + c.2.2.2

theorem confusionStep_total (c : Nat × Nat × Nat × Nat) (p a : Bool) :
    total (confusionStep c p a) = total c + 1 := by
  cases p <;> cases a <;> simp [confusionStep, total] <;> omega
