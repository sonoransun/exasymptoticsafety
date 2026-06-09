# Foliated EH 3D phase portrait

![Foliated EH 3D phase portrait](./foliated_3d.png)

3D phase portrait of the toolkit's schematic foliated Einstein–Hilbert truncation in `(g, lambda, lambda_ADM)` coupling space on `S^1 x S^3`. The highlighted plane `lambda_ADM = 1` is a fixed plane of the flow by construction (`beta_lambda_ADM ∝ g (lambda_ADM - 1)`), UV-repulsive at physical couplings. This truncation admits **no non-Gaussian fixed point with `g > 0`**: the only fixed point found — and marked here (circle) — is the *Gaussian* one at `g = lambda = 0` on the `lambda_ADM = 1` plane. The published foliated NGFP of Manrique et al. (2011) (Euclidean `g* ~ 0.19, lambda* ~ 0.31`, with `lambda_ADM = 1` imposed by their full-Diff ansatz) is a literature reference, not a root of this system.

## References

- Manrique, Rechenberger & Saueressig (2011), Phys. Rev. Lett. 106, 251302 [1003.5129].
- Biemans, Platania & Saueressig (2017), JHEP 05, 093 [1609.02803].
- Saueressig et al. (2025), Phys. Rev. D 111, 106007 [2501.03752].

## See in `docs/LITERATURE.md`

- [lorentzian-foliated](../LITERATURE.md#lorentzian-foliated)

## See also

- `asymsafety.gui.visualization_3d.foliated_phase_portrait_3d`
- `asymsafety.validation.manrique_2011`
