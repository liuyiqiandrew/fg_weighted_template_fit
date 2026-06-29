# Parallel Bootstrap Design

## Design Decision
`bootstrap_template_amplitudes` and
`bootstrap_template_amplitudes_multi_mask` use
`concurrent.futures.ThreadPoolExecutor` for opt-in parallel Monte Carlo draws
when `n_jobs > 1`. Threads were chosen over processes so the functions can be
called directly from scripts and JupyterLab cells without multiprocessing
entry-point guards, pickling local state, or copying large map arrays into child
processes.

The default remains `n_jobs=1`, which keeps the existing serial execution path
and random-number sequence.

## Function Interaction
Each Monte Carlo draw is independent:

1. draw target Q/U noise from `target_noise_cov`
2. draw optional noise for every `DifferenceTemplateInput` in the left,
   right-hand, and data-projection template stacks
3. rebuild noisy difference templates with `fit_foreground_templates` or,
   for multi-mask fits, `fit_foreground_templates_multi_mask`
4. call the weighted GLS solve
5. store the returned amplitude vector in `amplitude_samples[draw_index]`
   or the returned `(n_fit_mask, n_template)` block for multi-mask fits

In threaded mode, the shared inputs are read-only. Each worker constructs its
own noisy target and template inputs before fitting.

## Random Number Behavior
Serial mode uses the caller-provided generator exactly as before. Threaded mode
first draws one deterministic integer seed per Monte Carlo realization from the
caller-provided `rng`, then creates an independent `numpy.random.Generator` per
worker task. This avoids sharing mutable RNG state across threads.

Threaded results are reproducible for the same inputs, `rng`, and `n_jobs`, but
they are not required to match the exact serial sample sequence.

## Progress Behavior
`show_progress=True` still wraps the draw iterator with standard `tqdm` using
`total=n_mc`, `desc="Bootstrap MC"`, and `unit="draw"`. In threaded mode, the
progress bar advances as futures complete. This works in terminals and
JupyterLab output without requiring notebook widgets.

## Change Tracking
- Added `n_jobs` to `bootstrap_template_amplitudes` and
  `bootstrap_template_amplitudes_multi_mask`.
- Added threaded execution for `n_jobs > 1`.
- Added support for independent data-projection template stacks in bootstrap
  draws.
- Added tests for serial compatibility, threaded reproducibility, output shape,
  invalid worker counts, and threaded progress wrapping.
- Updated README and API documentation with the new option.
