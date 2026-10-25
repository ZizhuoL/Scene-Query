# Measured results

These measurements describe one TUM Freiburg desk scene on an Apple M2 Max / 64 GB. They are scene-specific engineering results, not an independent benchmark or a new-scene generalization claim. Runtime metadata for the six main comparisons is included in `reports/reconstruction_delivery.json`; scripts regenerate detailed reports and figures locally.

## Reconstruction

| Model | Training views | Median PSNR / dB | Median depth MAE / cm | Fixed quality floors |
| --- | ---: | ---: | ---: | --- |
| Neural field, RGB+depth, seed 42 | 48 | 18.845 | 4.577 | Fail |
| Neural field, RGB loss only, seed 42 | 48 | 17.334 | 14.268 | Fail |
| Neural field, RGB+depth, seed 43 | 48 | 18.623 | 4.776 | Fail |
| Gaussian, RGB only | 48 | 18.63 | 13.36 | Fail |
| Gaussian, opaque targets | 48 | 18.63 | 9.14 | Fail |
| Gaussian, dense observations | 525 | 21.84 | 7.72 | Pass |

All medians are across nine held-out views. The quality floors require median/worst PSNR >=20/17 dB and median depth MAE <=12 cm.

The dense fit ran 12,000 steps in 137.95 seconds; development selection chose its 10,000-step checkpoint. The selected model has 237,285 Gaussians, worst-view PSNR 18.09 dB, median SSIM 0.777 and worst per-view depth MAE 17.24 cm. The viewer displays a 220,000-point surface proxy. Reported child peak RSS was 0.684 GB; this excludes unmeasured GPU allocation and is not total memory consumption.

The three custom-field fits took 672.24 seconds in total, excluding setup and evaluation. The matched depth-loss comparison improved mean paired PSNR by 1.54 dB and reduced mean paired depth MAE by 17.53 cm. This is one control seed and nine correlated views: it does not establish statistical significance or broad reliability. The chosen field failed the color floors despite learning from random initialization. Two later development revisions also failed their quality guard.

The first Gaussian pilot failed its 20 dB development resource guard at 19.70 dB; a longer fit was still attempted. Opacity and input-density revisions followed. Later final views were already exposed by earlier attempts, so production success is repeated engineering validation. The production fit also uses more observations and cannot isolate an architecture advantage over the 48-view study.

## Search and interaction

The retained search checks returned 8/8 positive matches, 4/4 absent-object abstentions and 2/2 ambiguity responses on the small fixed query set. Query P95 latency was 1.52 seconds in the recorded regression run. All 14 statuses and point-ID lists were unchanged when the learned display geometry was integrated.

These counts use approximate project-specific reference regions, not independent human annotations. They describe this fitted scene and selected queries, not general accuracy. Text similarity remains uncalibrated. Search uses frozen OWLv2 and deterministic multi-view association, not trained semantic reasoning.

## Recovery and limits

A 40-step field recovery check compared uninterrupted training with stopping at step 20 and resuming model, optimizer and sampling state. Maximum parameter/RGB difference was 1.19e-7 and depth difference was 3.58e-7 m, with matching sampling RNG. MPS accumulation can still be nondeterministic. Brush parameter warm starts do not restore optimizer or sampler state.

Fine lettering, wires, reflective surfaces, incomplete geometry and occlusion boundaries remain difficult. Aggregate passing metrics do not guarantee every view is good. Dense-input gains combine data coverage and representation choices; the method does not discover hidden objects or reconstruct arbitrary new environments from a text query.
