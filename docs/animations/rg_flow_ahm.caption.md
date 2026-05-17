# RG flow into the AHM charged fixed point

[Watch the animation](./rg_flow_ahm.mp4)

RG trajectories in the `(u, r)` plane at fixed `α = α*` for the 3D
Abelian-Higgs model with `N_f = 60`, `N_c = 1`. The gauge direction
`α` is UV-repulsive at the charged FP; the animation therefore
projects to the critical surface — the `(u, r)` plane where the FP
attracts under forward RG time. Eight trajectories start on a small
ring around the FP and spiral in.

Regenerate with

```bash
python scripts/generate_animations.py --only rg_flow_ahm
```

(`--writer pillow` for `.gif` if `ffmpeg` is not on `PATH`.)

## References

- Bonati, Pelissetto & Vicari (2025), Phys. Rep. [arXiv:2410.05823].

## See also

- [`docs/cross-analogue-gauge-higgs.md`](../cross-analogue-gauge-higgs.md)
- `asymsafety.transforms.bridge.gauge_higgs.GaugeHiggsAnalogue`
- `asymsafety.visualization.animation.rg_trajectory_animation`
