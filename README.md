# SOFR / Treasury Spread Modeling

## Project Summary
This repository documents a full modeling progression for yield estimation on bond data using a benchmark-plus-spread framework.

It includes:
- early direct regression baselines,
- benchmark-aware spread models,
- issuer-level and bond-level decompositions,
- research upper-bound modeling,
- final no-leakage model.

The final clean result was a no-leakage bond-plus-issuer benchmark model with mean error around 5.8 bps and median around 3.7 bps.

## Core Formula
The final family of models used:

Y_i = B(T_i) + alpha_issuer(i) + beta_bond(i) + gamma^T x_i

where:
- Y_i = reported yield
- B(T_i) = benchmark Treasury curve at maturity T_i
- alpha_issuer(i) = issuer-level spread effect
- beta_bond(i) = bond-level spread effect
- x_i = residual features such as coupon, discount, and maturity

## Model Progression

### V6 — Standard baseline
Benchmark-plus-spread style model with issuer / feature structure.
- Good baseline
- Around 12 bps error
- Easy to explain

### V8 — Cleaner structured baseline
Uses:
- smooth benchmark curve,
- issuer effect,
- residual regression.
Still around 12 bps, but more defensible and better organized.

### V9 — Shrinkage test
Applied shrinkage to issuer effects.
- Error worsened
- Important finding: issuer signal was already strong

### V10 — Research upper bound
Adds bond-specific adjustments in-sample.
- Around 4–5 bps
- Too optimistic for production
- Useful as an upper bound

### V11 — Final clean model
Uses leave-one-out issuer and bond effects to reduce leakage.
- Around 5.8 bps mean
- Around 3.7 bps median
- Best defensible model

## Repository Structure
- v8_model.py — clean benchmark + issuer model
- v10_model.py — in-sample bond-level upper-bound model
- v11_model.py — final no-leakage research-grade model
- model_progression.md — paper-style methodology and model evolution

## Important Positioning
- V8 = production-style baseline
- V10 = research upper bound
- V11 = best clean model
