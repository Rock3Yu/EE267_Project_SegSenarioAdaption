from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from datasets import CarlaWeatherDataset
from eval import evaluate, evaluate_per_weather
from models import get_deeplabv3_resnet50
from utils.runner import (
    AverageMeter,
    create_optimizer,
    create_summary_writer,
    load_config,
    save_checkpoint,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CARLA weather baseline model")
    parser.add_argument("--cfg", required=True, help="YAML config file path")
    parser.add_argument("--device", default=None, help="GPU selection, e.g., cuda:0")
    return parser.parse_args()


def resolve_path(path_str: Optional[str], base_dir: Path) -> Optional[Path]:
    if path_str is None:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def build_dataset(cfg, split: str, is_train: bool):
    base_dir = Path(cfg["cfg_dir"])
    data_root = resolve_path(cfg.get("data_root", "./data"), base_dir)
    list_key = "train_list" if is_train else "val_list"
    list_path = resolve_path(cfg.get(list_key), base_dir)
    weather_key = "train_weathers" if is_train else "val_weathers"
    dataset = CarlaWeatherDataset(
        data_root=str(data_root),
        list_file=str(list_path) if list_path else None,
        weather_list=cfg.get(weather_key),
        image_size=cfg.get("image_size", (512, 512)),
        split=split,
        is_train=is_train,
        use_weather_aug=cfg.get("use_weather_aug", False),
    )
    return dataset


def build_dataloader(dataset, cfg, is_train: bool):
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 4)),
        shuffle=is_train,
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

    train_dataset = build_dataset(cfg, split="train", is_train=True)
    val_dataset = build_dataset(cfg, split="val", is_train=False)
    train_loader = build_dataloader(train_dataset, cfg, is_train=True)
    val_loader = build_dataloader(val_dataset, cfg, is_train=False)

    model = get_deeplabv3_resnet50(
        num_classes=int(cfg.get("num_classes", 20)),
        pretrained=bool(cfg.get("pretrained_backbone", True)),
        use_gn=bool(cfg.get("use_group_norm", True)),  # Default to GroupNorm for small batch sizes
    )
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=int(cfg.get("ignore_index", 255)))
    optimizer = create_optimizer(model, cfg)

    log_root = resolve_path(cfg.get("tensorboard_log_dir") or cfg.get("log_dir", "./logs"), Path(cfg["cfg_dir"]))
    writer = None
    if log_root is not None:
        try:
            writer = create_summary_writer(log_root, experiment_name=cfg.get("experiment_name", "baseline"))
        except Exception as exc:  # pragma: no cover - tensorboard optional
            print(f"TensorBoard initialization failed, skipping visualization: {exc}")

    epochs = int(cfg.get("epochs", 30))
    best_miou = 0.0
    save_path = resolve_path(cfg.get("save_path"), Path(cfg["cfg_dir"]))
    if save_path is None:
        save_dir = resolve_path(cfg.get("save_dir", "./logs/checkpoints"), Path(cfg["cfg_dir"]))
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{cfg.get('experiment_name', 'baseline')}.pth"

    try:
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_miou = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                num_classes=int(cfg.get("num_classes", 20)),
                ignore_index=int(cfg.get("ignore_index", 255)),
            )
            per_weather = None
            if cfg.get("eval_per_weather", False):
                per_weather = evaluate_per_weather(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    num_classes=int(cfg.get("num_classes", 20)),
                    ignore_index=int(cfg.get("ignore_index", 255)),
                )
            lr = optimizer.param_groups[0]["lr"]
            per_weather_str = ""
            if per_weather:
                items = [f"{k}={v:.4f}" for k, v in sorted(per_weather.items())]
                per_weather_str = " | per-weather: " + ", ".join(items)
            print(
                f"Epoch [{epoch}/{epochs}] - train_loss: {train_loss:.4f}, val_mIoU: {val_miou:.4f}{per_weather_str}",
                flush=True,
            )
            if writer:
                writer.add_scalar("train/loss", train_loss, epoch)
                writer.add_scalar("val/mIoU", val_miou, epoch)
                writer.add_scalar("lr", lr, epoch)
                if per_weather:
                    for weather, score in per_weather.items():
                        writer.add_scalar(f"val/mIoU_{weather}", score, epoch)
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
                print(f"  >> New best model saved to {save_path}")
    finally:
        if writer:
            writer.close()

    print(f"Training finished, best val mIoU = {best_miou:.4f}")


if __name__ == "__main__":
    main()
