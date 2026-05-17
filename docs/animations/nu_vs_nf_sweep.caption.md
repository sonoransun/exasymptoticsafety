# ν(N_f) sweep animation

[Watch the animation](./nu_vs_nf_sweep.mp4)

Animated companion to the static `nu_vs_nf.png` figure. As the red
marker sweeps `N_f` from 12 to 120, it traces out the toolkit's
one-loop 4-ε prediction for `ν` at the charged fixed point and lays
it against the large-`N_f` field-theory asymptote `ν = 1 − 9.727/N_f`
(grey dashed).

The pedagogical point is that the toolkit's perturbative scheme
approaches the Wilson-Fisher mean-field limit `ν → 1/2` from below,
while the field-theory asymptote (and the lattice Monte-Carlo
sample points in the static figure) approach `ν → 1` from below.
Both pictures agree that the charged FP exists; they disagree on the
quantitative value of `ν` because the perturbative 4-ε expansion is
far from convergent at `ε = 1`.

Regenerate with

```bash
python scripts/generate_animations.py --only nu_vs_nf_sweep
```

## References

- Bonati, Pelissetto & Vicari (2025), Phys. Rep. [arXiv:2410.05823].

## See also

- [`docs/cross-analogue-gauge-higgs.md`](../cross-analogue-gauge-higgs.md)
- `asymsafety.transforms.bridge.gauge_higgs.correlation_length_exponent`
- `asymsafety.visualization.animation.parameter_sweep_animation`
