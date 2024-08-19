# Codebase for investigating how GRUs store varaibles.

This is the full experimental setup used for our paper `Recurrent Neural Networks Learn to Store and Generate Sequences using Non-Linear Representations`. Below you can find a general description of the repository and how to use it.

## Goal

Provide an easy-to-use environment for NN architecture research. Supports many housekeeping tasks by default:
- Checkpoitning
- Using SLURM with W&B
- SLURM preemption
- Microbatching, length-bucketing
- LR scheduling
- Plotting on W&B and Tensorboard
- ...

## Structure
```
├───cache - temporary files automatically generated by this code
├───framework
│    ├─  datasets - a large collection of diverse datasets
│    ├─  visualize - universal plotting functions working for TF and W&B
│    ├─  helpers - helper routines for cluster, wandb and training setup
│    ├─  utils - useful utils (downloaders, samplers, multiprocessing)
│    ├─  layers - useful layers
│    ├─  tasks - main training loop, etc.
│    └─  * - all kinds of reusable components
│
├───save - saved checkpoints and trainig state
├───sweeps - Weights and Biases experiment configs
├───tasks - experiments. Add new experiments as new files here, it will be automatically picked up.
├───notebooks - Notebooks for exploration
└───main.py - initialization code
```

## Useful built-in arguments

- `-task`: which task to use. Tasks are picked up from tasks directory automatically. See how to create a new task in the `Creating a new task` chapter.
- `-name`: state will be saved in `save/<name>` folder.
- `-restore <checkpoint file>`: restores everything, including the command line arguments, from a checkpoint file. If any other argument is specified, it overwrites the one found in the checkpoint.
- `-reset 1`: do not load checkpoint from `save/<name>` but restart training.
- `-log`: can be `tb` for tensorboard or `wandb` for Weights & Biases. All supported plot types are defined in `framework/visualize/plot.py` and support logging on both. If `tb` is specified, the run will start a Tensorboard session on port 7000 (or the next available)
- `-gpu <index>`: which GPU to use. Leave empty for allocating the next empty one.
- `-lr <learning rate>`: specify learning rate
- `-batch_size <batch size>`: specify batch size
- `-wd`: weight decay
- `-stop_after <n_iters>`: terminate after this many iterations. It also sets the amount of steps for the LR scheduler if used.
- `-amp 1`: use mixed-precision training
- `-grad_clip <max norm>`: clip gradients to the this max norm. 1 by default. Specify `none` to disable.
- `-lr_sched.type cos`: use cos learning rate decay
- `-lr_warmup <n_iters>`: use linear LR warmup for this many steps.
- `-load_pretrained_model <checkpoint file>`: loads the model only, but not the arguments, opitmizer state, etc, from a checkpoint.
- `-length_bucketed_sampling 1`: groups examples of similar length into batches to save compute wasted for padding. Only works for some datasets.
- `-save_interval <n_iters>`: how often to save checkpoints.
- `-test_interval <n_iters>`: how often to run automatic tests.
- `-per_device_batch_size <batch size>`: specify the per-GPU batch size. Microbatching (gradient accumulation) will be used to ensure that the actual batch size is <= than the specified. Uneven division is supported.
- `-n_microbatch <number of microbatching steps>`: manually specify the number of microbatches. Mutually exclusive with `per_device_batch_size`.

There are many other useful default arguments, defined in `framework/task/task.py`, `framework/task/simple_task.py` and `framework/helpers/training_helper.py`.

## ClusterTool

The code is designed to work with [ClusterTool](https://github.com/RobertCsordas/cluster_tool).

If used wih ClusterTool, W&B sweeps, run preemption, file synchronization, etc will be handled automatically.

## Creating a new task

See `tasks/gru_repeat.py` for an example.

Add your file to the `tasks` directory. Use the following imports:
```python
from framework.task import task, args, SimpleTask
```

Inherit `SimpleTask` in your task, and decorate it with `@task()` decorator. You can define arguments by creating a function as follows:

```python
@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-state_size", default=128)
```

You will be able to access all defined arguments in `self.helper.args` within the task class.

In your training task, implement the following functions: `create_datasets()`, `create_model()` and `run_model(data: Dict[str, torch.Tensor], ubatch: int)` and `run_model_validation(data: Dict[str, torch.Tensor])`. It is safe to ignore the `ubatch` argument. For `run_model_validation`, the model has to return the loss and the result in a format required by the dataset. There is no documentation yet for this, sorry, but the general rule is if the dataset returns a field called `out_len`  then it should be a tuple of (output indices, output length), otherwise it should be logits.

`create_datasets()` should create `self.train_set` and an arbitrary number of validation sets in `self.valid_sets`.

See the implementation of `Task` (`framework/task/task.py`) and `SimpleTask` (`framework/task/simple_task.py`) to see how to implement custom validaiton and special functionality.

## General policy
- Whenever possible keep changes outside `framework`. If somehting is generally useful, let me know and we will move it to `framework`.
- The code should be self contained. If a dataset should be downloaded or generated, the code should figure it automatically whenever it is possible (sometimes the data is behind login, like for DM Math, then it's ok to download it manually).
- External dependencies should be minimalized.
- Final experiments, or long exepriments should be self contained Use notebooks only for exploration.