# SceneQuery

Explore a reconstructed 3D desk scene, search for objects with text, and inspect the source photographs behind each match. SceneQuery runs locally and combines a trained scene representation with frozen open-vocabulary detection and multi-view geometric association.

## Features

- Text queries such as `keyboard`, `telephone`, `laptop` and `game controller` select regions anchored in 3D.
- Orbit the scene, switch candidates, inspect original photographs and adopt a source camera's viewpoint.
- Broad queries expose ambiguity; unsupported queries such as `elephant` can return no match.
- Compare held-out renders from a custom neural field and a locally trained Gaussian reconstruction.

The interactive display is a point proxy of the learned surface. It is not a live Gaussian rasterizer. The public scene is TUM RGB-D `freiburg1/desk`; no personal photographs are required.

## Requirements

The measured environment was Apple M2 Max / 64 GB, macOS 15.7.9, Python 3.12.3, PyTorch 2.6.0 and Node 24.7.0. Neural training requires Metal access from a native macOS terminal. The Gaussian reference evaluator requires the macOS C++ compiler. Python and JavaScript dependencies are pinned.

The repository contains source and configuration, not datasets, model weights or generated scene assets. Setup downloads the public inputs; subsequent processing and inference are local.

## Setup and reconstruction

Run these commands from the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --cache-dir cache/pip -r requirements.lock.txt
npm ci --cache cache/npm --ignore-scripts
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/download_model.py
.venv/bin/python scripts/prepare.py
.venv/bin/python scripts/train.py
.venv/bin/python scripts/cache_semantics.py
.venv/bin/python scripts/evaluate.py --run-id reproduction
.venv/bin/python scripts/download_brush.py
```

Create an isolated reconstruction workspace, then train and evaluate the models:

```sh
.venv/bin/python scripts/start_reproduction.py --name my-reproduction
cd artifacts/reproductions/my-reproduction
make train-neural evaluate-neural train-brush evaluate-brush
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/verify_artifacts.py
./run.sh
```

Open **http://127.0.0.1:8765**. Keep using this reproduction directory when restarting the app. Press Ctrl+C to stop the server.

The reproduction workspace shares the prepared data, environment, frozen detector and base surfel assets, while keeping reconstruction outputs separate. Do not run base data/model/surfel preparation inside it. Use a new reproduction name for another experiment. Training and evaluation can take longer than the reported model-fit times, which exclude data setup, semantic caching and rendering.

For an interrupted field run, use the same configuration and `--resume`, for example:

```sh
.venv/bin/python scripts/train_radiance.py --config config/neural/rgbd_s42.json --resume
```

Field checkpoints preserve model, optimizer and sampling state. Brush PLY checkpoints support parameter warm starts, not exact optimizer recovery. GPU results need not be bitwise identical across runs.

## Results and method

The dense Gaussian fit achieved **21.84 dB median / 18.09 dB worst-view PSNR** and **7.72 cm median per-view depth MAE** across nine held-out views. Its 12,000-step optimization took about **138 seconds** on the measured Mac. The custom neural field and sparse Gaussian variants failed the fixed color-quality requirements; those outcomes are included in the comparison.

See [METHOD.md](METHOD.md) for the learning and geometry contract, [RESULTS.md](RESULTS.md) for measurements and limitations, and [ATTRIBUTION.md](ATTRIBUTION.md) for data, model and software attribution.

## Repository layout

| Path | Purpose |
| --- | --- |
| `scenequery/`, `web/` | Query backend, geometry/rendering code and browser interface |
| `scripts/`, `tests/` | Preparation, training, evaluation and correctness checks |
| `config/`, `manifests/` | Fixed configurations, input identities and release source hashes |
| `reports/` | Reproduction integrity specifications and measured app metadata |
| `licenses/` | Required third-party notices |

Generated data, checkpoints, media, dependency installations and runtime reports remain local. The Git ignore file uses an explicit source-file allowlist; new files require deliberate inclusion.

## Limitations

This is one fitted scene with supplied poses and depth, not new-scene generalization. Search scores are not calibrated confidence, and the small evaluation uses approximate project-specific references rather than independent human ground truth. Thin structures, reflective surfaces and occlusion boundaries remain difficult. The system does not discover hidden objects or support arbitrary relational reasoning. See the results document before interpreting aggregate metrics.
