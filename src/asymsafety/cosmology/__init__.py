"""RG-improved classical solutions for asymptotic safety.

Couples ``BetaFunctionSystem`` solutions (fixed points and trajectories)
to classical gravity by promoting Newton's constant G and the cosmological
constant Λ to scale-dependent quantities G(k(x)) and Λ(k(x)).

Submodules:
    scale_identification: Maps physical scales (r, t, H) to RG scales k.
    rg_improved_bh: RG-improved Schwarzschild (Bonanno-Reuter 2000) and
        related static spherically symmetric solutions.
    rg_improved_flrw: Friedmann-Lemaître-Robertson-Walker cosmologies with
        running G(t), Λ(t).

References:
    Bonanno & Reuter (2000), Phys. Rev. D 62, 043008 [hep-th/0002196]
    Reuter & Saueressig (2012), New J. Phys. 14, 055022 [1202.2274]
    Platania (2020), Eur. Phys. J. C 80, 470 (review of AS phenomenology)
"""
