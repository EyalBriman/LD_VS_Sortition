# Consolidation and correction notes

This file records the main changes made while turning the supplied standalone scripts into one project.

## Correctness fixes

1. **Correct clique target sampling.**
   The loss-bound script used `argmax(W / U)` with uniform random variables. That is not a categorical draw with probabilities proportional to `W`. The consolidated implementation uses row-wise categorical inverse-CDF sampling. The SBM implementation uses the equivalent exponential-race construction `argmin(E / W)`.

2. **Correct global singleton conditioning.**
   The singleton distribution is estimated as
   `sum_graph q_raw_i / sum_graph P(M=1 | graph)`, which is the required ratio of expectations.

3. **More accurate acceptance-rate diagnostics.**
   Rejection sampling now counts every accepted draw in the final batch when estimating `P(M=1)`, even if only part of that batch is needed to reach the requested sample count.

4. **Explicit cycle diagnostics in Experiment 2.**
   Representative-free delegation cycles were implicit in the original chain-following code. They are now counted and written to the output data.

5. **Safe plotting with incomplete parameter grids.**
   Plotting functions build matching x and y arrays, so custom parameter subsets do not create length mismatches.

6. **Callable compact Experiment 2 plot.**
   The compact-summary function is defined before the experiment runner uses it.

## Structural improvements

- Shared geometry, aggregation, delegation, and selection modules.
- One CLI for all simulations.
- Quick and full profiles.
- JSON configuration saved with every run.
- Draw-level CSV output and diagnostic summaries.
- Headless figure generation.
- Automated tests and smoke tests.
- Windows Git Bash and batch helpers.
