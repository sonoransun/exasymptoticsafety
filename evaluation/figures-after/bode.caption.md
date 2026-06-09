# Bode plot of the EH stability matrix

![Bode plot of the EH stability matrix](./bode.png)

Magnitude (in dB) and phase (in degrees) of the (0, 0) entry of the impedance transfer function `H(s) = (sI - M)^{-1}` at the Reuter NGFP. Poles in the right half-plane correspond to UV-attractive (relevant) directions; their location on the imaginary axis sets the dominant resonance frequencies.

## References

- Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030].
- Trefethen & Embree (2005), *Spectra and Pseudospectra*.

## See in `docs/LITERATURE.md`

- [reviews](../LITERATURE.md#reviews)

## See also

- `asymsafety.transforms.bridge.impedance.ImpedanceBridge`
- `asymsafety.transforms.visualization.transform_plots.plot_bode`
