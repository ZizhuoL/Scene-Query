# Method

## Inputs and coordinates

TUM RGB-D `freiburg1/desk` supplies registered 640 x 480 RGB/depth frames and camera poses. RGB/depth timestamps are associated within 20 ms; translation is interpolated and rotation uses quaternion SLERP. Sixty deterministic views provide 48 training views, three development views and nine final views.

Camera axes are x right, y down, z forward. Poses map camera coordinates into world coordinates; depth PNG values are divided by 5000. The registered-depth path uses fx=fy=525, cx=319.5 and cy=239.5. Resizing updates projection consistently. Input hashes and camera transforms are retained in `manifests/`.

## Geometry and appearance

The base representation initializes colored points from training RGB-D, keeps one observation per 8 mm voxel and caps the set at 120,000 points. Adam then optimizes XYZ and RGB for 240 steps using sampled photometric/depth losses and a position anchor. Camera parameters remain fixed. Initialization and fusion are deterministic processing; the gradient optimization is training.

The custom neural field starts from random parameters. Its 530,116 trainable parameters comprise multi-resolution plane features, a shared decoder, density and view-dependent color heads. Gradients flow through volume rendering from RGB and training-depth losses. Three matched 6,000-step runs compare RGB+depth seed 42, RGB-loss-only seed 42 and RGB+depth seed 43. The seed-42 pair shares initialization and sampling settings. Bounds use training depth, so the control is a depth-loss ablation, not an entirely depth-free method. Rendering inference uses only camera rays and learned parameters.

The Gaussian reconstruction uses Brush v0.3.0. Trainable means, scales, rotations, opacity and degree-2 spherical-harmonic appearance are fitted locally. Training RGB-D initializes points; RGB/alpha losses optimize the model. Depth is not a Brush training loss. The production fit uses 525 observations outside a +/-0.10-second exclusion interval around every development/final image. It has denser inputs than the 48-view comparison and represents a different interpolation task.

Development views determine checkpoint selection. Fixed final floors are median PSNR >=20 dB, worst-view PSNR >=17 dB and median per-view depth MAE <=12 cm. Camera-z expected depth includes residual transmittance at 3.5 m. Analytic geometry tests and comparison with native development renders check the CPU Gaussian evaluator.

## Text-to-region association

Google OWLv2 supplies frozen image and text encoders. Cached image features and boxes come only from training views. Query text is encoded using the checkpoint's fixed photo template; no semantic module is trained.

Candidate source boxes are refined with GrabCut and projected onto the fitted surfel anchors. Points must agree with source depth and receive consistent multi-view support. Connected regions become candidates with stable world-space point IDs. The viewer highlights these IDs while the camera moves and shows supporting original photos.

The query engine uses training-source RGB-D for visibility and provenance at inference. Held-out images and reference boxes never enter fitting, semantic caching or query scoring. Multiple supported regions expose ambiguity, and insufficient evidence produces abstention. Similarity is not a probability of correctness.

## Evaluation boundaries

Search checks use eight approximate reference boxes on held-out photographs, four absent-object queries and two broad queries. The references are project-specific, not independent human ground truth. They were prepared separately from the semantic detector; the small sample cannot establish benchmark accuracy. Later checks reused previously inspected views and are engineering validation, not an untouched confirmatory test.

The neural field, sparse Gaussian attempts and later production fit have different failure modes. Results are reported separately, including failures. No new-scene, hidden-object or general spatial-reasoning capability is claimed.

## Reproducibility

Input and model releases are pinned by hashes. Integrity specifications under `reports/` validate the source/configuration used for a new run; measured-source identities are retained where distributed error-message wording differs. These changes do not alter numerical methods. Fresh reproduction results may differ slightly from recorded measurements.

The application serves explicit local asset routes through FastAPI and displays depth-tested point geometry through Three.js. Training-source surfels remain the search anchors when switching to the Gaussian surface proxy. The evidence panel uses precomputed held-out renders; it does not render Gaussians live.
