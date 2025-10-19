import os
from typing import List

import pydantic
import torch
import torch.distributed as dist
import yaml
from omegaconf import OmegaConf

from pretrain import (
    PretrainConfig,
    create_dataloader,
    create_evaluators,
    evaluate,
    init_train_state,
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)


class EvalConfig(pydantic.BaseModel):
    checkpoint: str

    save_outputs: List[str] = [
        "inputs",
        "labels",
        "puzzle_identifiers",
        "logits",
        "q_halt_logits",
        "q_continue_logits",
    ]


def launch():
    eval_cfg = EvalConfig(**OmegaConf.to_container(OmegaConf.from_cli()))  # type: ignore

    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ and DEVICE == "cuda":
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK
            and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )
    with open(
        os.path.join(os.path.dirname(eval_cfg.checkpoint), "all_config.yaml"), "r"
    ) as f:
        config = PretrainConfig(**yaml.safe_load(f))

        config.eval_save_outputs = eval_cfg.save_outputs
        config.checkpoint_path = os.path.dirname(eval_cfg.checkpoint)

    # Dataloader
    _, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    eval_loader, eval_metadata = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Models
    train_state = init_train_state(
        config, train_metadata, rank=RANK, world_size=WORLD_SIZE
    )
    # Try unwrap torch.compile
    try:
        train_state.model.load_state_dict(
            torch.load(eval_cfg.checkpoint, map_location="cuda"), assign=True
        )
    except:
        train_state.model.load_state_dict(
            {
                k.removeprefix("_orig_mod."): v
                for k, v in torch.load(eval_cfg.checkpoint, map_location="cuda").items()
            },
            assign=True,
        )

    train_state.step = 0
    ckpt_filename = os.path.basename(eval_cfg.checkpoint)
    if ckpt_filename.startswith("step_"):
        train_state.step = int(ckpt_filename.removeprefix("step_"))

    # Evaluate
    print("Starting evaluation")

    train_state.model.eval()
    metrics = evaluate(
        config,
        train_state,
        eval_loader,
        eval_metadata,
        evaluators,
        rank=RANK,
        world_size=WORLD_SIZE,
        cpu_group=CPU_PROCESS_GROUP,
    )

    if metrics is not None:
        print(metrics)


if __name__ == "__main__":
    launch()
