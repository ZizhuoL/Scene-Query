# Attribution and use terms

## Scene data

TUM RGB-D Benchmark, `rgbd_dataset_freiburg1_desk`, official release last modified 2011-09-30. Dataset use terms: **Creative Commons Attribution 4.0 International (CC BY 4.0)**, unless otherwise stated by the source. Official license statement: https://cvg.cit.tum.de/data/datasets/rgbd-dataset#license.

Citation: Jürgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, Daniel Cremers. *A Benchmark for the Evaluation of RGB-D SLAM Systems.* IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2012.

Changes: deterministic frame selection, timestamp association, pose interpolation, depth backprojection, locally optimized XYZ/RGB, semantic association and visualized projections. Original source photographs remain unmodified in the photo dialog. Raw media and all derived scene/media files are excluded from Git. Preserve TUM attribution and CC BY 4.0 when sharing derived media; this checkout does not publish any assets.

## Models and methods

Release: Google **OWLv2**, `google/owlv2-base-patch16-ensemble`, pinned revision in `config/model.json`. Model card declares **Apache-2.0**: https://huggingface.co/google/owlv2-base-patch16-ensemble. Matthias Minderer, Alexey Gritsenko, Neil Houlsby. *Scaling Open-Vocabulary Object Detection*, NeurIPS 2023. https://arxiv.org/abs/2306.09683.

Initial investigated model: Google **OWL-ViT**, `google/owlvit-base-patch32`, revision `cbc355fb364588351c5d51c7f74465e8e7ec6f72`; model card declares Apache-2.0. Minderer et al. *Simple Open-Vocabulary Object Detection with Vision Transformers*, ECCV 2022. https://arxiv.org/abs/2205.06230. Initial failures are retained in the evaluation reports. Neither detector was trained or fine-tuned by this project.

Original search representation: depth-supervised colored surfels with standard Adam optimization and differentiable image sampling. GrabCut performs foreground refinement; graph connected components and depth consistency perform deterministic multi-view association. The original surfel stage does not implement LangSplat, LERF or Brush. The later locally trained Brush reconstruction is attributed below. No new detector or reconstruction algorithm is claimed.

## Software

- PyTorch and torchvision: BSD-style licenses, https://github.com/pytorch/pytorch and https://github.com/pytorch/vision.
- Hugging Face transformers and huggingface_hub: Apache-2.0, https://github.com/huggingface/transformers and https://github.com/huggingface/huggingface_hub.
- Three.js: MIT, https://github.com/mrdoob/three. Installed license is retained with the local npm dependency.
- OpenCV: Apache-2.0, https://github.com/opencv/opencv.
- NumPy / SciPy: BSD-3-Clause, https://numpy.org and https://scipy.org.
- Pillow: HPND, https://python-pillow.org.
- FastAPI / Starlette / Uvicorn: MIT / BSD-3-Clause / BSD-3-Clause, respectively; source links and licenses are included in installed package metadata.

Code licenses do not determine the separate dataset or checkpoint terms. Dependency versions are pinned in the lockfiles. This is a local implementation; no third-party weights or media are committed to Git.

## Neural reconstruction continuation

The scene-specific neural field is randomly initialized and trained in this project; no additional pretrained reconstruction checkpoint or dataset is used. The implementation draws on established ideas: Mildenhall et al., NeRF (ECCV 2020), for differentiable volume rendering; Chen et al., TensoRF (ECCV 2022), for compact spatial factorization; Deng et al., DS-NeRF (CVPR 2022), for depth-constrained ray termination. It is not presented as their exact architecture or a reproduction of their benchmark results. Method and experimental differences are specified in NEURAL_RECONSTRUCTION.md.

PyTorch 2.6.0 supplies autograd and the Metal backend. Matplotlib 3.10.8 generates standalone experimental figures. Exact Python dependencies are pinned in requirements.lock.txt. Existing TUM data, OWLv2 weights and software terms remain distinct; the new checkpoints and rendered media stay in ignored artifacts/.

The established tri-plane representation is also described by Chan et al., EG3D ([project page](https://nvlabs.github.io/eg3d/)). SceneQuery directly optimizes plane features for one scene and does not use its GAN, pretrained generator, images or source code.


## Brush Gaussian reconstruction

The production scene and two sparse-input comparisons use [Brush v0.3.0](https://github.com/ArthurBrussee/brush/releases/tag/v0.3.0), source commit `3edecbb2fe79d3e2c87eeab85b15e0b1dd10d486`, by Arthur Brussee and contributors, under Apache-2.0. The license is retained in `licenses/BRUSH-LICENSE.txt`. The official Apple silicon release checksum, binary checksum, complete commands and data transformations are pinned in `reports/brush/`. No pretrained Brush scene or weights are imported. Scene parameters are optimized locally from the TUM training observations.

`scenequery/gaussian.py` and `gaussian_blend.cpp` implement a CPU reference evaluator using standard Gaussian covariance projection, spherical harmonics and alpha compositing. The pinned Brush shader source defines the projection/compositing conventions against which this implementation is checked; its license is retained. Agreement with native development renders is measured, rather than assumed. Brush uses the established 3D Gaussian splatting method; see [Kerbl et al., 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). No new Gaussian-rendering algorithm or reproduction of upstream benchmark scores is claimed.
