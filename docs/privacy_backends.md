# Optional privacy research backends

VertiMosaic v0.2 adds an optional Gaussian release mechanism with a zCDP accountant.

`GaussianDPBackend` requires the caller to provide an L2 sensitivity bound. It adds Gaussian noise with standard deviation `noise_multiplier * l2_sensitivity` and composes releases using zCDP:

- one release contributes `rho = 1 / (2 * noise_multiplier^2)`;
- repeated releases add `rho`; and
- the accountant converts composed `rho` to an `(epsilon, delta)` bound.

This is a **mechanism-level research backend**. It is not silently enabled in VFL training and must not be described as end-to-end differential privacy unless every data-dependent release in a concrete protocol is covered by a valid sensitivity bound and accounting argument.

The backend registry also names PSI, secure aggregation, MPC and homomorphic encryption as planned extension points. Those mechanisms remain unimplemented and therefore remain outside VertiMosaic's guarantees.
