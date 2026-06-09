# Regulator shape functions

![Regulator shape functions](./regulator_comparison.png)

Comparison of the Litim (optimised, step-function) and Exponential (smooth) IR regulators. The Litim regulator yields closed-form beta functions and minimises scheme dependence; the Exponential regulator is `C^infty` but requires numerical integration.

## References

- Wetterich (1993), Phys. Lett. B 301, 90.
- Litim (2001), Phys. Rev. D 64, 105007 [hep-th/0103195].

## See also

- `asymsafety.visualization.phase_portrait.regulator_comparison_panels` — phase-portrait-level scheme comparison.
