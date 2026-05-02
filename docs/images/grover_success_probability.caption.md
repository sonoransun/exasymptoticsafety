# Grover success probability vs iteration count

![Grover success probability vs iteration count](./grover_success_probability.png)

Theoretical Grover success probability `P(k) = sin^2((2k+1) arcsin sqrt(M/N))` for an EH coupling grid, with the optimal iteration count `k_opt` highlighted. The monotonic-then-oscillatory shape is the diagnostic signature that justifies the sqrt(N) speedup.

## References

- Grover (1996), [quant-ph/9605043].

## See also

- `asymsafety.quantum.grover.search.GroverFixedPointSearch`
