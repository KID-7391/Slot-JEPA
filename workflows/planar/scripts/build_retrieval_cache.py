"""Build a planar frame-slot cache and TMP retrieval bank on all GPUs."""

from pathlib import Path
import sys

import hydra
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf, open_dict


SCRIPTS_ROOT = Path(__file__).resolve().parent
PLANAR_ROOT = SCRIPTS_ROOT.parent
for path in (SCRIPTS_ROOT, PLANAR_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import custom_resolvers  # noqa: F401,E402
from plan import (  # noqa: E402
    _load_runtime_model_config,
    _resolve_recorded_retrieval_bank,
    load_model,
)
from train import Trainer  # noqa: E402
from utils import cfg_to_dict, seed  # noqa: E402


@hydra.main(version_base=None, config_path="../conf", config_name="plan")
def main(cfg):
    cfg_dict = cfg_to_dict(cfg, resolve=True)
    accelerator = Accelerator()
    device = accelerator.device
    launch_cwd = Path(hydra.utils.get_original_cwd())

    ckpt_base_path = Path(cfg_dict["ckpt_base_path"])
    if not ckpt_base_path.is_absolute():
        ckpt_base_path = (launch_cwd / ckpt_base_path).resolve()
    model_root = ckpt_base_path / "outputs" / cfg_dict["model_name"]
    model_ckpt = (
        model_root / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    )
    config_candidates = (
        model_root / "hydra.yaml",
        model_root / ".hydra" / "config.yaml",
    )
    model_config_path = next(
        (path for path in config_candidates if path.exists()), None
    )
    if model_config_path is None:
        raise FileNotFoundError(f"Could not find training config under {model_root}")

    checkpoint_payload = torch.load(model_ckpt, map_location="cpu")
    recorded_path = checkpoint_payload.get("retrieval_bank_path")
    if recorded_path is not None:
        packaged_bank = _resolve_recorded_retrieval_bank(
            model_ckpt, recorded_path
        )
        if packaged_bank is not None:
            if accelerator.is_main_process:
                print(f"Retrieval bank already exists: {packaged_bank}")
            accelerator.wait_for_everyone()
            return

    model_cfg = _load_runtime_model_config(model_config_path, launch_cwd)
    if str(
        model_cfg.model.get("predictor_architecture", "transition_memory")
    ) == "sequential":
        if accelerator.is_main_process:
            print("Sequential predictor does not use a retrieval bank")
        accelerator.wait_for_everyone()
        return
    seed(cfg_dict["seed"])
    sample_dsets, trajectory_dsets = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
        random_transition_sampling=False,
        causal_transition_sampling=True,
    )

    cache_root = Path(cfg_dict.get("inference_cache_dir", launch_cwd / "slot-ac"))
    if not cache_root.is_absolute():
        cache_root = (launch_cwd / cache_root).resolve()
    build_cfg = OmegaConf.create(
        OmegaConf.to_container(model_cfg, resolve=False)
    )
    with open_dict(build_cfg):
        build_cfg.ckpt_base_path = str(cache_root)

    builder = Trainer.__new__(Trainer)
    builder.cfg = build_cfg
    builder.datasets = {"train": sample_dsets["train"]}
    builder.validation_dataset_keys = []
    builder.device = device
    builder.is_slot_model = True
    builder.accelerator = accelerator

    slot_cache_path = builder._slot_cache_file_path("train")
    bank_path = builder._retrieval_bank_file_path(
        slot_cache_path, sample_dsets["train"]
    )
    if bank_path.exists():
        if accelerator.is_main_process:
            print(f"Retrieval bank already exists: {bank_path}")
        accelerator.wait_for_everyone()
        return
    if not bool(cfg_dict.get("auto_build_retrieval_bank", True)):
        raise FileNotFoundError(
            f"Retrieval bank is missing and auto-build is disabled: {bank_path}"
        )

    if accelerator.is_main_process:
        print(
            "Building retrieval cache from raw trajectories with "
            f"{accelerator.num_processes} GPU process(es): {bank_path}"
        )
    model = load_model(
        model_ckpt,
        model_cfg,
        model_cfg.num_action_repeat,
        device=device,
        dset=trajectory_dsets["valid"],
        load_retrieval_bank=False,
        log_checkpoint_load=accelerator.is_main_process,
        checkpoint_payload=checkpoint_payload,
    )
    builder.model = model
    # Building only: planning workers load the finished bank after this
    # distributed job exits, so do not replicate the large bank onto every GPU.
    builder.prepare_slot_cache_if_needed(load_retrieval_bank=False)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"Retrieval cache ready: {bank_path}")


if __name__ == "__main__":
    main()
