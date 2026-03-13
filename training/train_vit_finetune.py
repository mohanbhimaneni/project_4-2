import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class DicomDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        image_size: int = 224,
        augment: bool = False,
        skip_compressed: bool = True,
        augment_intensity: float = 0.15,
        augment_gamma: float = 0.15,
        augment_noise_std: float = 0.02,
        augment_translate: int = 8,
    ):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment
        self.skip_compressed = skip_compressed
        self.augment_intensity = max(0.0, float(augment_intensity))
        self.augment_gamma = max(0.0, float(augment_gamma))
        self.augment_noise_std = max(0.0, float(augment_noise_std))
        self.augment_translate = max(0, int(augment_translate))

    @staticmethod
    def _normalize(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        img -= img.min()
        vmax = img.max()
        if vmax > 0:
            img /= vmax
        return img

    @staticmethod
    def _to_2d(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img

        if img.ndim == 3:
            if img.shape[-1] <= 4:
                return img.mean(axis=-1)

            if img.shape[0] <= 4:
                return img.mean(axis=0)

            mid = img.shape[0] // 2
            return img[mid]

        img = np.squeeze(img)
        if img.ndim > 2:
            first_axis = img.shape[0] // 2
            img = img[first_axis]

        if img.ndim != 2:
            raise ValueError(f"Unsupported DICOM pixel shape after squeeze: {img.shape}")

        return img

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augmentations(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
        if random.random() < 0.2:
            x = torch.flip(x, dims=[1])

        if self.augment_intensity > 0:
            scale = 1.0 + random.uniform(-self.augment_intensity, self.augment_intensity)
            shift = random.uniform(-0.5 * self.augment_intensity, 0.5 * self.augment_intensity)
            x = torch.clamp(x * scale + shift, 0.0, 1.0)

        if self.augment_gamma > 0 and random.random() < 0.5:
            gamma_min = max(0.5, 1.0 - self.augment_gamma)
            gamma_max = 1.0 + self.augment_gamma
            gamma = random.uniform(gamma_min, gamma_max)
            x = torch.clamp(x, 0.0, 1.0) ** gamma

        if self.augment_noise_std > 0 and random.random() < 0.5:
            noise_std = random.uniform(0.0, self.augment_noise_std)
            x = torch.clamp(x + torch.randn_like(x) * noise_std, 0.0, 1.0)

        if self.augment_translate > 0 and random.random() < 0.5:
            shift_y = random.randint(-self.augment_translate, self.augment_translate)
            shift_x = random.randint(-self.augment_translate, self.augment_translate)
            x = torch.roll(x, shifts=(shift_y, shift_x), dims=(1, 2))

        return x

    def __getitem__(self, idx: int):
        dcm_path, label = self.samples[idx]
        try:
            ds = pydicom.dcmread(str(dcm_path))
            ts_uid = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
            if self.skip_compressed and ts_uid is not None and getattr(ts_uid, "is_compressed", False):
                raise RuntimeError("Compressed transfer syntax skipped for robust training")
            img = ds.pixel_array
            img = self._to_2d(img)
            img = self._normalize(img)
        except Exception:
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = x.squeeze(0)

        if self.augment:
            x = self._apply_augmentations(x)

        x = x.repeat(3, 1, 1)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path_value: str, repo_root: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p

    # First honor current working directory semantics.
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    # Fallback to repository-root-relative resolution.
    repo_candidate = (repo_root / p).resolve()
    return repo_candidate


def load_overview_binary_samples(overview_csv: Path, dicom_dir: Path) -> Tuple[List[Tuple[Path, int]], Dict[str, int]]:
    samples: List[Tuple[Path, int]] = []
    with overview_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dcm_name = row["dicom_name"].strip()
            contrast = row["Contrast"].strip().lower() == "true"
            dcm_path = dicom_dir / dcm_name
            if dcm_path.exists():
                samples.append((dcm_path, int(contrast)))

    if not samples:
        raise RuntimeError("No samples found. Check --overview-csv and --dicom-dir.")

    return samples, {"False": 0, "True": 1}


def _iter_dcm_files(folder_path: Path, images_per_series: int) -> List[Path]:
    files = sorted(folder_path.glob("*.dcm"))
    if images_per_series > 0:
        files = files[:images_per_series]
    return files


def load_manifest_samples(
    metadata_csv: Path,
    dataset_root: Path,
    label_column: str,
    images_per_series: int,
    min_samples_per_class: int,
    excluded_labels: List[str],
) -> Tuple[List[Tuple[Path, int]], Dict[str, int]]:
    grouped: Dict[str, List[Path]] = {}

    with metadata_csv.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_value = row.get(label_column, "").strip()
            if not label_value:
                continue
            if label_value in excluded_labels:
                continue

            rel_loc = row.get("File Location", "").strip()
            if not rel_loc:
                continue

            rel_loc = rel_loc.replace(".\\", "").replace("\\", "/")
            folder_path = dataset_root / rel_loc
            if not folder_path.exists():
                continue

            dcm_files = _iter_dcm_files(folder_path, images_per_series=images_per_series)
            if not dcm_files:
                continue

            grouped.setdefault(label_value, []).extend(dcm_files)

    if not grouped:
        raise RuntimeError("No manifest samples found. Check --metadata-csv and --dataset-root.")

    filtered = {k: v for k, v in grouped.items() if len(v) >= min_samples_per_class}
    if not filtered:
        raise RuntimeError(
            "All classes were filtered out by --min-samples-per-class. "
            "Lower this threshold or verify data."
        )

    labels_sorted = sorted(filtered.keys())
    label_to_id = {label: idx for idx, label in enumerate(labels_sorted)}

    samples: List[Tuple[Path, int]] = []
    for label_name, files in filtered.items():
        class_id = label_to_id[label_name]
        for file_path in files:
            samples.append((file_path, class_id))

    return samples, label_to_id


def evaluate(model, loader, device, num_classes: int):
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_targets.extend(targets.detach().cpu().numpy().tolist())

    metrics = {
        "acc": accuracy_score(all_targets, all_preds),
        "f1_macro": f1_score(all_targets, all_preds, average="macro"),
    }

    try:
        prob_arr = np.array(all_probs)
        if num_classes == 2:
            metrics["auc"] = roc_auc_score(all_targets, prob_arr[:, 1])
        else:
            metrics["auc"] = roc_auc_score(all_targets, prob_arr, multi_class="ovr", average="macro")
    except Exception:
        metrics["auc"] = float("nan")

    return metrics


def build_class_weights(
    train_samples: List[Tuple[Path, int]],
    num_classes: int,
    mode: str,
) -> Optional[torch.Tensor]:
    if mode == "none":
        return None

    counts = np.zeros(num_classes, dtype=np.float64)
    for _, label in train_samples:
        counts[label] += 1.0

    counts = np.maximum(counts, 1.0)

    if mode == "inverse":
        weights = 1.0 / counts
    elif mode == "sqrt_inverse":
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unsupported class-weighting mode: {mode}")

    weights = weights / max(weights.mean(), 1e-12)
    return torch.tensor(weights, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser("Fine-tune ViT on DICOM datasets (overview CSV or manifest metadata CSV)")
    parser.add_argument("--dataset-type", type=str, default="manifest", choices=["manifest", "overview"])

    parser.add_argument("--overview-csv", type=str, default="dataset/siim-medical-images/versions/6/overview.csv")
    parser.add_argument("--dicom-dir", type=str, default="dataset/siim-medical-images/versions/6/dicom_dir")

    parser.add_argument("--metadata-csv", type=str, default="dataset/manifest-1740445452889/metadata.csv")
    parser.add_argument("--dataset-root", type=str, default="dataset/manifest-1740445452889")
    parser.add_argument("--label-column", type=str, default="Modality")
    parser.add_argument(
        "--exclude-labels",
        type=str,
        default="SR",
        help="Comma-separated label values to exclude from training (e.g. SR)",
    )
    parser.add_argument("--images-per-series", type=int, default=1)
    parser.add_argument("--min-samples-per-class", type=int, default=2)
    parser.add_argument("--skip-compressed", action="store_true", default=True)
    parser.add_argument("--allow-compressed", action="store_true", default=False)

    parser.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="project_understanding/checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--class-weighting",
        type=str,
        default="inverse",
        choices=["none", "inverse", "sqrt_inverse"],
        help="Class weighting mode for CrossEntropyLoss",
    )
    parser.add_argument("--augment-intensity", type=float, default=0.15)
    parser.add_argument("--augment-gamma", type=float, default=0.15)
    parser.add_argument("--augment-noise-std", type=float, default=0.02)
    parser.add_argument("--augment-translate", type=int, default=8)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = resolve_path(args.outdir, repo_root)
    outdir.mkdir(parents=True, exist_ok=True)

    overview_csv_path = resolve_path(args.overview_csv, repo_root)
    dicom_dir_path = resolve_path(args.dicom_dir, repo_root)
    metadata_csv_path = resolve_path(args.metadata_csv, repo_root)
    dataset_root_path = resolve_path(args.dataset_root, repo_root)

    if args.dataset_type == "overview":
        if not overview_csv_path.exists():
            raise FileNotFoundError(
                f"overview CSV not found: {overview_csv_path} (cwd={Path.cwd()})"
            )
        if not dicom_dir_path.exists():
            raise FileNotFoundError(
                f"DICOM directory not found: {dicom_dir_path} (cwd={Path.cwd()})"
            )
    else:
        if not metadata_csv_path.exists():
            raise FileNotFoundError(
                f"metadata CSV not found: {metadata_csv_path} (cwd={Path.cwd()})"
            )
        if not dataset_root_path.exists():
            raise FileNotFoundError(
                f"dataset root not found: {dataset_root_path} (cwd={Path.cwd()})"
            )

    if args.dataset_type == "overview":
        samples, label_to_id = load_overview_binary_samples(overview_csv_path, dicom_dir_path)
    else:
        samples, label_to_id = load_manifest_samples(
            metadata_csv=metadata_csv_path,
            dataset_root=dataset_root_path,
            label_column=args.label_column,
            images_per_series=args.images_per_series,
            min_samples_per_class=args.min_samples_per_class,
            excluded_labels=[x.strip() for x in args.exclude_labels.split(",") if x.strip()],
        )

    labels = [label for _, label in samples]
    num_classes = len(label_to_id)

    if num_classes < 2:
        raise RuntimeError("Need at least 2 classes for classification.")

    train_samples, val_samples = train_test_split(
        samples,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=labels,
    )

    skip_compressed = args.skip_compressed and not args.allow_compressed
    train_ds = DicomDataset(
        train_samples,
        image_size=args.image_size,
        augment=True,
        skip_compressed=skip_compressed,
        augment_intensity=args.augment_intensity,
        augment_gamma=args.augment_gamma,
        augment_noise_std=args.augment_noise_std,
        augment_translate=args.augment_translate,
    )
    val_ds = DicomDataset(
        val_samples,
        image_size=args.image_size,
        augment=False,
        skip_compressed=skip_compressed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = timm.create_model(args.model_name, pretrained=True, num_classes=num_classes).to(device)

    class_weights = build_class_weights(train_samples, num_classes, mode=args.class_weighting)
    if class_weights is not None:
        print(f"Using class weighting mode '{args.class_weighting}' with weights: {class_weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        print("Using unweighted CrossEntropyLoss")
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    id_to_label = {v: k for k, v in label_to_id.items()}
    print(f"Dataset type: {args.dataset_type}")
    print(f"Total samples: {len(samples)} | train: {len(train_samples)} | val: {len(val_samples)}")
    print(f"Classes ({num_classes}): {id_to_label}")

    best_auc = -1.0
    start_epoch = 1

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        print(f"Resuming from checkpoint: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location="cpu")

        resume_model_name = resume_ckpt.get("model_name")
        if resume_model_name and resume_model_name != args.model_name:
            raise RuntimeError(
                f"Resume checkpoint model_name={resume_model_name} does not match --model-name={args.model_name}"
            )

        resume_num_classes = int(resume_ckpt.get("num_classes", num_classes))
        if resume_num_classes != num_classes:
            raise RuntimeError(
                f"Resume checkpoint num_classes={resume_num_classes} does not match current dataset num_classes={num_classes}"
            )

        model.load_state_dict(resume_ckpt["model_state_dict"], strict=True)

        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        if torch.cuda.is_available() and resume_ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])

        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        prev_auc = float(resume_ckpt.get("val_metrics", {}).get("auc", float("nan")))
        if not np.isnan(prev_auc):
            best_auc = prev_auc

        print(f"Resume start epoch: {start_epoch} | best_auc={best_auc:.4f}")

    if start_epoch > args.epochs:
        print(
            f"Checkpoint already at epoch {start_epoch - 1}, which is >= --epochs ({args.epochs}). "
            "Nothing to train."
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()

        if epoch <= args.freeze_epochs:
            for name, p in model.named_parameters():
                p.requires_grad = name.startswith("head")
        elif epoch == args.freeze_epochs + 1:
            for p in model.parameters():
                p.requires_grad = True

        train_loss = 0.0
        n_train = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            n_train += batch_size

        scheduler.step()
        train_loss = train_loss / max(n_train, 1)

        val_metrics = evaluate(model, val_loader, device, num_classes=num_classes)
        val_auc = val_metrics["auc"]

        state = {
            "epoch": epoch,
            "model_name": args.model_name,
            "num_classes": num_classes,
            "label_to_id": label_to_id,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if torch.cuda.is_available() else None,
            "val_metrics": val_metrics,
            "args": vars(args),
        }

        torch.save(state, outdir / "last.pt")

        if (not np.isnan(val_auc)) and (val_auc > best_auc):
            best_auc = val_auc
            torch.save(state, outdir / "best.pt")

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"val_f1_macro={val_metrics['f1_macro']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f}"
        )

    print(f"Training complete. Best AUC: {best_auc:.4f}")
    print(f"Saved checkpoints in: {outdir}")


if __name__ == "__main__":
    main()
