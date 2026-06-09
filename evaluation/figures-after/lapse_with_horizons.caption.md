# RG-improved Schwarzschild lapse with horizons

![RG-improved Schwarzschild lapse with horizons](./lapse_with_horizons.png)

Lapse `f(r) = 1 - 2 G(r) M / r` for a super-critical mass `M = 8` on a trajectory with IR Newton constant `G_N = 0.02` (Planck mass `M_pl = G_N^{-1/2} ~ 7.07`, critical mass `M_crit ~ 5.7`). The running `G(r) = g(k(r))/k(r)^2` is constant in the IR and softens like `g* r^2` inside the Planck radius, so both Bonanno-Reuter horizons appear: the inner Cauchy horizon `r_-` and the outer event horizon `r_+` (dashed vertical lines). As `r -> 0` the lapse returns to 1 only *linearly* with the `k = 1/r` cutoff used here: the central singularity is softened (curvature still diverges, but much more mildly than classically), not replaced by a regular de Sitter core.

## References

- Bonanno & Reuter (2000), Phys. Rev. D 62, 043008 [hep-th/0002196].
- Platania (2023), in *Handbook of Quantum Gravity* [2302.04272].

## See in `docs/LITERATURE.md`

- [black-holes-cosmology](../LITERATURE.md#black-holes-cosmology)

## See also

- `asymsafety.cosmology.visualization.plot_lapse_with_horizons`
