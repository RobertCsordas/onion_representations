import os

from typing import Optional
import framework
from framework.task import task_db
import torch
import json
from framework import dataset
import tasks

torch.backends.cudnn.benchmark = True


def register_args(parser: framework.helpers.ArgumentParser):
    task_db.register_args(parser)
    parser.add_argument("-task", default="tuple")
    parser.add_argument("-load_pretrained_model", type=str)
    parser.add_argument("-test_pretrained", default=True)
    parser.add_argument("-test_only", default=False)


def initialize(restore: Optional[str] = None):
    helper = framework.helpers.TrainingHelper(wandb_project_name="lm",
                                              register_args=register_args,
                                              extra_dirs=["export", "model_weights", "tmp"],
                                              log_async=True, restore=restore)

    task = task_db.get_task(helper.args.task)

    task = task(helper)

    if helper.args.load_pretrained_model:
        print("Loading pretrained model...")

        pretrained = os.path.expanduser(helper.args.load_pretrained_model)
        if not helper.args.load_pretrained_model.endswith(".pth"):
            pretrained = os.path.join(pretrained, str(helper.args.sweep_id_for_grid_search), "model.pth")

        assert os.path.isfile(pretrained), f"Failed to load pretrained weights. File {pretrained} not found."

        if helper.dist_env.is_master():
            task.load_weights(pretrained)

        helper.distibute_model_weights()
        print("Done.")

    return helper, task


def main():
    helper, task = initialize()



    if helper.args.test_only:
        res = task.validate()
        helper.log(res)
        print("Validate returned:")
        print(json.dumps(res))
        print("-------------------")
    else:
        if helper.args.test_pretrained and helper.args.load_pretrained_model:
            helper.log({f"load_validation/{k}": v for k, v in task.validate().items()})

        task.train()

    task.finish()
    helper.finish()


if __name__ == "__main__":
    main()
