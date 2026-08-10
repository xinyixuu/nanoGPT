# 3D digit-token trajectory viewer

This viewer follows the single-file Three.js report style in this directory,
including OrbitControls, canvas labels, a dark explanatory panel, and keyboard
controls. Unlike a dimensionality-reduction visualization, every plotted point
is the model's actual three-component token embedding.

Run `demos/digits_3d_trajectory_demo.sh` from the repository root to generate
`token_trajectories.json`. Then serve the repository with
`python3 -m http.server 8000` and visit
`http://localhost:8000/report/threejs/digits-3d/index.html`.

Orange points are trained digit-like tokens. Blue points are configurable
letters that exist in the vocabulary and model parameters but never occur in
either dataset split. Use the slider, arrow keys, or Space playback to compare
their trajectories over checkpoint time.

By default all points remain on the radius-`sqrt(3)` sphere, so the viewer
isolates directional motion rather than changes in vector magnitude. Run the
demo with `WTE_FIXED_NORM=false` to produce the earlier unconstrained view.

The viewer also accepts `?data=<relative-json-path>`. The sweep demo uses this
to keep every vocabulary-size and fixed-norm variation under `runs/` rather
than overwriting the default trajectory.
