# Contributing Workflow

## tl;dr:

1. **PR Early; Merge Often:** create PR as soon as possible, and merge as soon as ready (e.g. not crashing
   other features, works, and passing in CI).
2. **Briefly say what it is; show how to use it:** add "demo" script in `demos/` folder and small markdown doc in `./documentation` with how to run the demo
3. **Add a Test:** add or augment existing "test" script so our "CI" can make sure no one else breaks it.

## Philosophy: _Pull Early_; _Merge Often_

Due to the rapid pace of development in this repository, it is best to post PR's
even __before they are ready__ for early feedback, and merge your features __as
soon as they are complete__.

## Documentation and Demo Scripts

### tl;dr:

Procedure:
2. documentation file (./documentation/)
1. demo script (how to test the feature)
3. [bonus] modify or create an "exploration"

### Docs

To facilitate pace of development, we provide a "documentation/" directory, but
keep it lightweight.

For new variation files, create a doc with their name e.g. `AttentionVariations.md`

Then add a section describing what the contribution is, why it is interesting
(one-liner), add a link if there is a paper associated with it.

### Demos

Small file showing how to use it (e.g. if it is a variation, go to "add exploration")

### Adding an "Exploration"

Explorations are yaml files which support a grid search for comparisons.

These are very appropriate for comparison of variations, e.g. optimizer
variations where we can compare a list of optimizers on the same dataset and
number of iterations.

Each variation file should have at least one exploration covering each.

See `../explorations/sample.yaml` for a template

## Tests

All our features (as much as possible) should be covered by a test that ensures
they work.

This prevents future changes from breaking existing contributions.

### Process

To ensure key contributions/variations are maintained:
1. add a test script to "tests/" with minimal size network to test functionality
2. loop into an existing `.github/workflows` file, or create a new GitHub CI workflows file, to safeguard your feature.

### Test Directories and Templates:

- `.github/workflows/` - all CI tests will be here, feel free to copy any there as a template
- `tests/` - create a bash script here which will go through the full procedure to ensure the feature still works.

### Keep tests small to ensure they run on Github CI's Cpus

These these tests will need to be on __cpu__ (github ci lacks gpus), you must
scale down the network to ensure any test will run at reasonable speeds.

### More the merrier

Tests all run parallel -- the more the merrier!

These tests are run by the Github CI Runner, and test whenever anyone:
- posts a PR
- commits to a standing PR


## Pull Request Guide:

0. Create a fork of the repo via GitHub (will only have to do this once)

- i. Log into your account on github, and navigate to  https://github.com/ReaLLMASIC/nanoGPT
- ii. (optional but recommended) star this repo : D
- iii. fork a copy of this repo (only needs to be done once)

1. Push your feature to your fork of the repo:

Add clone main repo, but add your fork of the repo as a remote:
```sh
git clone https://github.com/ReaLLMASIC/nanoGPT
git remote add <your_username> https://github.com/your_username/nanoGPT
```

2. Push to your fork:
```bash
git push <your_username> main:<name_of_feature>
```

3. Navigate to Github to create a pull request to the ReaLLMASIC/nanoGPT repo.


