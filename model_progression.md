# Model Progression and Mathematical Walkthrough

## Problem Setup
The direct-yield approach was not sufficient. The model was restructured as:

Yield = BenchmarkRate + Spread

where spread absorbs issuer and bond-specific credit differences.

## Benchmark construction
A benchmark curve was loaded from treasury_curve[Confidential CSV], then smoothed with monotone cubic interpolation (PCHIP):

B(T) = PCHIP(T)

This provides a smooth Treasury benchmark at any maturity T.

## Spread definition
S_i = Y_i - B(T_i)

where:
- Y_i is the observed yield,
- B(T_i) is the benchmark Treasury rate at bond maturity,
- S_i is the benchmark-relative spread.

## V8
Y_i = B(T_i) + alpha_issuer(i) + gamma^T x_i

### Interpretation
- benchmark gives the base rate,
- issuer captures credit structure,
- residual regression explains leftover shape.

## V9
Shrinkage was introduced on the issuer term:
alpha_i_shrunk = [n_i / (n_i + lambda)] * issuer_mean_i + [lambda / (n_i + lambda)] * global_mean

This worsened results, showing issuer effects were already informative.

## V10
Y_i = B(T_i) + alpha_issuer(i) + beta_bond(i) + gamma^T x_i

with bond effects estimated in-sample.

### Interpretation
This is an upper bound, not a production result, because each bond helps estimate its own adjustment.

## V11
Same structure as V10, but issuer and bond effects are estimated with leave-one-out logic:
- current row excluded from issuer mean
- current row excluded from bond mean

This makes V11 the strongest defensible result.

## Key empirical takeaway
Once benchmark, issuer, and bond structure are included, the linear feature coefficients become very small. That suggests most of the explanatory power comes from structural decomposition rather than simple regression alone.
