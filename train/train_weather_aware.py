from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from datasets import CarlaWeatherDataset
from eval import evaluate
from models import get_deeplabv3_resnet50
from samplers import build_weather_balanced_sampler
from utils.runner import (
    AverageMeter,
    create_optimizer,
    create_summary_writer,
    load_config,
    save_checkpoint,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser(description="天气感知训练脚本")
    parser.add_argument("--cfg", required=True, help="YAML配置文件")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve_path(path_str: Optional[str], base_dir: Path) -> Optional[Path]:
    if path_str is None:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def build_dataset(cfg, split: str, is_train: bool, weather_list: Optional[List[str]] = None):
    base_dir = Path(cfg["cfg_dir"])
    data_root = resolve_path(cfg.get("data_root", "./data"), base_dir)
    list_key = "train_list" if is_train else "val_list"
    list_path = resolve_path(cfg.get(list_key), base_dir)
    dataset = CarlaWeatherDataset(
        data_root=str(data_root),
        list_file=str(list_path) if list_path else None,
        weather_list=weather_list or cfg.get("train_weathers" if is_train else "val_weathers"),
        image_size=cfg.get("image_size", (512, 512)),
        split=split,
        is_train=is_train,
        use_weather_aug=cfg.get("use_weather_aug", False),
    )
    return dataset


def build_curriculum(cfg) -> List[Dict]:
    if "curriculum" in cfg:
        return cfg["curriculum"]
    epochs = int(cfg.get("epochs", 30))
    sampler_weights = cfg.get("sampler_weights") or {}
    def_stage = [
        {
            "name": "stage_clear",
            "epochs": cfg.get("stage1_epochs") or max(1, int(0.4 * epochs)),
            "weathers": cfg.get("stage1_weathers") or ["clear"],
            "weight_map": cfg.get("stage1_weights") or {"clear": 1.0},
        },
        {
            "name": "stage_rain",
            "epochs": cfg.get("stage2_epochs") or max(1, int(0.3 * epochs)),
            "weathers": cfg.get("stage2_weathers") or ["clear", "rain"],
            "weight_map": cfg.get("stage2_weights") or {"clear": 1.0, "rain": 2.0},
        },
        {
            "name": "stage_night",
            "epochs": cfg.get("stage3_epochs") or (epochs),
            "weathers": cfg.get("stage3_weathers") or ["clear", "rain", "night"],
            "weight_map": cfg.get("stage3_weights") or {"clear": 1.0, "rain": 2.0, "night": 3.0},
        },
    ]
    for stage in def_stage:
        for weather in stage["weathers"]:
            if weather in sampler_weights:
                stage["weight_map"][weather] = sampler_weights[weather]

    total = sum(stage["epochs"] for stage in def_stage)
    if total < epochs:
        def_stage[-1]["epochs"] += epochs - total
    return def_stage


def build_dataloader(cfg, dataset, sampler=None, shuffle: bool = True):
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 4)),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter("train_loss")
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)["out"]
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=images.size(0))
    return loss_meter.avg


def main():
    args = parse_args()
    cfg = load_config(args.cfg)
    seed_everything(int(cfg.get("seed", 42)))
    device_str = args.device or cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    curriculum = build_curriculum(cfg)
    full_train_weathers = cfg.get("all_weathers") or cfg.get("train_weathers")
    train_dataset = build_dataset(cfg, split="train", is_train=True, weather_list=full_train_weathers)
    val_dataset = build_dataset(cfg, split="val", is_train=False)

    def rebuild_loader(stage_cfg):
        train_dataset.set_active_weathers(stage_cfg["weathers"])
        train_dataset.enable_weather_aug(bool(cfg.get("use_weather_aug", True)))
        sampler = None
        if cfg.get("use_balanced_sampler", True):
            sampler = build_weather_balanced_sampler(train_dataset, stage_cfg.get("weight_map"))
        return build_dataloader(cfg, train_dataset, sampler=sampler, shuffle=not bool(sampler))

    train_loader = rebuild_loader(curriculum[0])
    val_loader = build_dataloader(cfg, val_dataset, sampler=None, shuffle=False)

    model = get_deeplabv3_resnet50(
        num_classes=int(cfg.get("num_classes", 20)),
        pretrained=bool(cfg.get("pretrained_backbone", True)),
    )
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=int(cfg.get("ignore_index", 255)))
    optimizer = create_optimizer(model, cfg)

    log_root = resolve_path(cfg.get("tensorboard_log_dir") or cfg.get("log_dir", "./logs"), Path(cfg["cfg_dir"]))
    writer = None
    if log_root is not None:
        try:
            writer = create_summary_writer(log_root, experiment_name=cfg.get("experiment_name", "weather_aware"))
        except Exception as exc:  # pragma: no cover
            print(f"TensorBoard 初始化失败，将跳过可视化: {exc}")

    epochs = int(cfg.get("epochs", 30))
    best_miou = 0.0
    save_path = resolve_path(cfg.get("save_path"), Path(cfg["cfg_dir"]))
    if save_path is None:
        save_dir = resolve_path(cfg.get("save_dir", "./logs/checkpoints"), Path(cfg["cfg_dir"]))
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{cfg.get('experiment_name', 'weather_aware')}.pth"

    stage_ptr = 0
    stage_end_epoch = curriculum[0]["epochs"]

    try:
        for epoch in range(1, epochs + 1):
            if epoch > stage_end_epoch and stage_ptr + 1 < len(curriculum):
                stage_ptr += 1
                stage_end_epoch += curriculum[stage_ptr]["epochs"]
                train_loader = rebuild_loader(curriculum[stage_ptr])
                print(f"进入 {curriculum[stage_ptr]['name']}，天气={curriculum[stage_ptr]['weathers']}")

            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_miou = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                num_classes=int(cfg.get("num_classes", 20)),
                ignore_index=int(cfg.get("ignore_index", 255)),
            )
            lr = optimizer.param_groups[0]["lr"]
            current_stage = curriculum[stage_ptr]["name"]
            print(
                f"Epoch [{epoch}/{epochs}] - stage={current_stage} loss={train_loss:.4f} val mIoU={val_miou:.4f}",
                flush=True,
            )
            if writer:
                writer.add_scalar("train/loss", train_loss, epoch)
                writer.add_scalar("val/mIoU", val_miou, epoch)
                writer.add_scalar("train/lr", lr, epoch)
                writer.add_scalar("train/stage_index", stage_ptr, epoch)
                writer.flush()
            if val_miou > best_miou:
                best_miou = val_miou
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_miou": best_miou,
                        "config": cfg,
                    },
                    save_path,
                )
                print(f"  >> 新最佳模型已保存至 {save_path}")
    finally:
        if writer:
            writer.close()

    print(f"训练结束，最佳 val mIoU = {best_miou:.4f}")


if __name__ == "__main__":
    main()
