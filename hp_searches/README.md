# Hyperparameter Searches

This is a folder for hyperparameter searches, currenlty we have a greedy search
algorithm which allows us to inspect the balance of parameters when growing a
model from a baseline.

## Usage

1. Create a baseline.yaml file and store in the hp_searches/ dir
2. Create a bash script for running this and store in the hp_searches/ dir
3. Run bash script from main directory
4. View with `python view_hp_log.py results.yaml`
5. Monitor via the above, and update max_iters as necessary.

## Why Grow a Model?

We find that we have a large number of hyperparameters, and the hyperparameters
have a strong interdependency on each other.

This means that we really should continually test compatbility of different
techniques during the search.

One way the team is exploring to do this, is to start a small model, and
continually take a step in the most parameter efficient direction.

In this case, we can ensure compatbility of the different parameters (as they
are continually co-tested), and each step will be the greatest increase in
capability per parameter.

## Limitations of Target Optimization

Our prior attempts at optimization include:

1. Optimizing hyperparameters for set # of total parameters:
    a. the hyperparameter space was still too large
    b. we had a danger that the model would simply increase its number of parameters to fit in the maximum space, making it hard to interpret the ultimate shape.

2. Optimizing for parameter efficiency
    a. the most efficient parameters for % of next token correctness were the initial parameters, so this would tend our models to go to zero-parameters.

## How Growing the Model Addresses 1 and 2

*Continual Pressure for Parameter Efficiency*

In this way we provide, "pressure" for the model to balance its parameters at
each stage of growth.

Essentially we have a gradient at each of the steps for how to grow the model,
and try to normalize by the number of additional parameters required to obtain
that improvement (the most cost efficient step).

There are other types of cost efficiency we can also explore, as well as
specific task targets, that make this an interesting framework to develop.

## Supported Metrics and Planned Extensions

Currently the only supported metric is validation loss or rather 1/exp(val_loss)
which is equal to the probability that the next token is correct.

Later we hope to also include summarization tasks, translation tasks, etc. so
and see what model shapes result from each of these.

Also the "Cost" can be altered as well.

## Notes And Observations

### Mitigating Step Noise:

There are a couple ways to improve step noise:

1. Increase the training data, e.g. max_iters (stay less than 1 epoch)
2. Increase the # random iterations to average (e.g. if data is limited)

### Step Settings Considerations:

1. Too large a step size and we can miss sub-optimizations
2. Too small a step size and we encounter too much noise
3. Increase iteration number -- try multiples of the step size, hoping to get
   past noise. Too night an iteration number is wasteful due to reduction in the
   % correct naturally for parameter even in the right direction.

### Training Step Considerations

1. too many training steps, too minimal delta param.
2. too few training steps, too much noise

There are some "humps" for learned parameters, e.g. absolute position encodings,
which typically resolve after 10,000 iterations or so. Currently we try to set a
at 5000 iteration count (with eval_interval at 5000) then move to 20_000 once we
hit a snag.


## Next Steps

### Means to increase maxiters automatically upon no feasible direction
We might program a means to have this occur automatically (step size for
max_iters specifically) if we have negative or zero for each of the parameters
directions

### Curve Fitting on Ratios

With the resulting data, we can try to find the curve of parametesr per mlp
size, nad vest val loss vs different characteristics, and ratios of differnet
ones via curve fitting. This could yield insights on balancing parameters
especially towards small language models.
