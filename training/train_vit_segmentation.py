import argparse
import csv
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


@dataclass
class PairRow:
    subject_id: str
    study_uid: str
    session: str
    split: str
    ct_dir: Path
    seg_file: Path


@dataclass
class SliceSample:
    subject_id: str
    split: str
    ct_dir: Path
    seg_file: Path
    z_index: int
    has_fg: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_training_device(preferred: str = "auto") -> Tuple[torch.device, str]:
    choice = preferred.lower().strip()

    if choice in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda"), "cuda"

    if choice == "directml":
        return torch.device("cpu"), "cpu"

    if choice == "cpu" or choice == "auto":
        return torch.device("cpu"), "cpu"

    if choice == "cuda":
        raise RuntimeError("--device cuda requested but CUDA is not available.")

    raise ValueError(f"Unsupported device choice: {preferred}")


def resolve_path(path_value: str, repo_root: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p

    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (repo_root / p).resolve()


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    preds = (torch.sigmoid(logits) > 0.5).float()
    targets = targets.float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return float(dice.mean().item())


def soft_dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    inter = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def combined_segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice_loss = soft_dice_loss_from_logits(logits, targets)
    total = bce_weight * bce + (1.0 - bce_weight) * dice_loss
    return total, bce, dice_loss


def bce_metric_on_cpu(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits_cpu = logits.detach().float().cpu()
    targets_cpu = targets.detach().float().cpu()
    return F.binary_cross_entropy_with_logits(logits_cpu, targets_cpu)


def load_pair_rows(manifest_csv: Path) -> List[PairRow]:
    rows: List[PairRow] = []
    with manifest_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ct_dir = Path(row["ct_dir"])
            seg_file = Path(row["seg_file"])
            if not ct_dir.exists() or not seg_file.exists():
                continue
            rows.append(
                PairRow(
                    subject_id=row["subject_id"],
                    study_uid=row["study_uid"],
                    session=row.get("session", "unknown"),
                    split=row["split"],
                    ct_dir=ct_dir,
                    seg_file=seg_file,
                )
            )

    if not rows:
        raise RuntimeError("No usable rows in manifest.")
    return rows


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _read_ct_series(ct_dir: Path) -> Tuple[List[pydicom.dataset.FileDataset], Dict[str, int]]:
    dcm_files = sorted(ct_dir.glob("*.dcm"))
    if not dcm_files:
        raise RuntimeError(f"No DICOM files in CT dir: {ct_dir}")

    slices: List[pydicom.dataset.FileDataset] = []
    for fp in dcm_files:
        ds = pydicom.dcmread(str(fp), stop_before_pixels=False)
        if not hasattr(ds, "PixelData"):
            continue
        slices.append(ds)

    if not slices:
        raise RuntimeError(f"No pixel slices in CT dir: {ct_dir}")

    def sort_key(ds):
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            return _safe_float(ipp[2], 0.0)
        return _safe_float(getattr(ds, "InstanceNumber", 0), 0.0)

    slices = sorted(slices, key=sort_key)
    sop_to_idx = {str(ds.SOPInstanceUID): i for i, ds in enumerate(slices) if hasattr(ds, "SOPInstanceUID")}
    return slices, sop_to_idx


def _normalize_hu(arr: np.ndarray, ds: pydicom.dataset.FileDataset, hu_min: int, hu_max: int) -> np.ndarray:
    slope = _safe_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
    intercept = _safe_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
    hu = arr.astype(np.float32) * slope + intercept
    hu = np.clip(hu, hu_min, hu_max)
    hu = (hu - hu_min) / max(1.0, float(hu_max - hu_min))
    return hu.astype(np.float32)


def _extract_referenced_sop_uid(frame_fg) -> Optional[str]:
    try:
        deriv = frame_fg.DerivationImageSequence
        if not deriv:
            return None
        src = deriv[0].SourceImageSequence
        if not src:
            return None
        return str(src[0].ReferencedSOPInstanceUID)
    except Exception:
        return None


def _read_seg_volume(seg_file: Path, sop_to_idx: Dict[str, int], ct_shape_hw: Tuple[int, int], n_slices: int) -> np.ndarray:
    ds = pydicom.dcmread(str(seg_file), stop_before_pixels=False)

    modality = str(getattr(ds, "Modality", "")).upper()
    if modality != "SEG":
        raise RuntimeError(f"Expected SEG modality, got {modality} for {seg_file}")

    pix = ds.pixel_array
    if pix.ndim == 2:
        pix = pix[None, ...]
    pix = (pix > 0).astype(np.uint8)

    if not hasattr(ds, "PerFrameFunctionalGroupsSequence"):
        raise RuntimeError(f"SEG missing PerFrameFunctionalGroupsSequence: {seg_file}")

    h, w = ct_shape_hw
    volume = np.zeros((n_slices, h, w), dtype=np.uint8)

    n_frames = min(len(ds.PerFrameFunctionalGroupsSequence), pix.shape[0])
    for i in range(n_frames):
        frame_fg = ds.PerFrameFunctionalGroupsSequence[i]
        sop_uid = _extract_referenced_sop_uid(frame_fg)
        if sop_uid is None:
            continue

        z = sop_to_idx.get(sop_uid)
        if z is None:
            continue

        frame_mask = pix[i]
        if frame_mask.shape != (h, w):
            frame_t = torch.from_numpy(frame_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            frame_t = F.interpolate(frame_t, size=(h, w), mode="nearest")
            frame_mask = frame_t.squeeze(0).squeeze(0).numpy().astype(np.uint8)

        volume[z] = np.maximum(volume[z], frame_mask)

    return volume


def build_slice_samples(
    rows: List[PairRow],
    split_name: str,
    pos_only: bool,
    max_bg_per_case: int,
    seed: int,
) -> List[SliceSample]:
    selected = [r for r in rows if r.split == split_name]
    rng = random.Random(seed)

    out: List[SliceSample] = []
    for row in selected:
        try:
            ct_slices, sop_to_idx = _read_ct_series(row.ct_dir)
            hw = ct_slices[0].pixel_array.shape
            seg_vol = _read_seg_volume(row.seg_file, sop_to_idx, hw, len(ct_slices))
        except Exception:
            continue

        pos_idx = [i for i in range(seg_vol.shape[0]) if np.any(seg_vol[i] > 0)]
        if not pos_idx:
            continue

        if pos_only:
            chosen = pos_idx
        else:
            all_idx = list(range(seg_vol.shape[0]))
            neg_idx = [i for i in all_idx if i not in set(pos_idx)]
            rng.shuffle(neg_idx)
            neg_idx = neg_idx[:max_bg_per_case]
            chosen = pos_idx + neg_idx

        for z in chosen:
            has_fg = z in set(pos_idx)
            out.append(
                SliceSample(
                    subject_id=row.subject_id,
                    split=row.split,
                    ct_dir=row.ct_dir,
                    seg_file=row.seg_file,
                    z_index=z,
                    has_fg=has_fg,
                )
            )

    return out


class RiderSegSliceDataset(Dataset):
    def __init__(
        self,
        samples: List[SliceSample],
        image_size: int = 224,
        hu_min: int = -1000,
        hu_max: int = 400,
        augment: bool = False,
    ):
        self.samples = samples
        self.image_size = image_size
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.augment = augment

        self._pair_cache: Dict[Tuple[str, str], Tuple[List[pydicom.dataset.FileDataset], np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pair_cache(self, sample: SliceSample) -> Tuple[List[pydicom.dataset.FileDataset], np.ndarray]:
        key = (str(sample.ct_dir), str(sample.seg_file))
        if key in self._pair_cache:
            return self._pair_cache[key]

        ct_slices, sop_to_idx = _read_ct_series(sample.ct_dir)
        hw = ct_slices[0].pixel_array.shape
        seg_vol = _read_seg_volume(sample.seg_file, sop_to_idx, hw, len(ct_slices))

        self._pair_cache[key] = (ct_slices, seg_vol)
        return ct_slices, seg_vol

    def _augment(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])
            mask = torch.flip(mask, dims=[2])
        if random.random() < 0.2:
            img = torch.flip(img, dims=[1])
            mask = torch.flip(mask, dims=[1])
        return img, mask

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        ct_slices, seg_vol = self._load_pair_cache(sample)

        z = sample.z_index
        ds = ct_slices[z]
        ct_img = _normalize_hu(ds.pixel_array, ds, self.hu_min, self.hu_max)
        seg_img = seg_vol[z].astype(np.float32)

        x = torch.from_numpy(ct_img).unsqueeze(0).unsqueeze(0)
        y = torch.from_numpy(seg_img).unsqueeze(0).unsqueeze(0)

        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        y = F.interpolate(y, size=(self.image_size, self.image_size), mode="nearest")

        x = x.squeeze(0)
        y = y.squeeze(0)

        if self.augment:
            x, y = self._augment(x, y)

        x = x.repeat(3, 1, 1)
        y = (y > 0.5).float()
        return x, y


class ViTSegHead(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="")
        embed_dim = self.encoder.num_features

        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def _tokens_to_map(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.ndim != 3:
            raise RuntimeError(f"Expected [B, N, C] features, got shape={tuple(feats.shape)}")

        if feats.shape[1] > 1:
            feats = feats[:, 1:, :]

        b, n, c = feats.shape
        side = int(np.sqrt(n))
        if side * side != n:
            raise RuntimeError(f"Patch token count {n} is not a perfect square")

        fmap = feats.reshape(b, side, side, c).permute(0, 3, 1, 2).contiguous()
        return fmap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder.forward_features(x)
        fmap = self._tokens_to_map(feats)
        logits = self.decoder(fmap)
        logits = F.interpolate(logits, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
        return logits


def evaluate(
    model,
    loader,
    device,
    use_inference_mode: bool = True,
    train_bce_weight: float = 0.5,
    device_backend: str = "cpu",
) -> Dict[str, float]:
    model.eval()
    bce_meter = 0.0
    dice_meter = 0.0
    n = 0

    criterion = nn.BCEWithLogitsLoss()

    eval_context = torch.inference_mode() if use_inference_mode else torch.no_grad()
    with eval_context:
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            if device_backend == "directml" or train_bce_weight <= 0.0:
                loss = bce_metric_on_cpu(logits, masks)
            else:
                loss = criterion(logits, masks)
            dice = dice_score_from_logits(logits, masks)

            bs = images.size(0)
            bce_meter += float(loss.item()) * bs
            dice_meter += dice * bs
            n += bs

    return {
        "bce": bce_meter / max(n, 1),
        "dice": dice_meter / max(n, 1),
    }


def build_balanced_weights(samples: List[SliceSample], pos_weight: float) -> torch.Tensor:
    safe_pos_weight = max(1.0, float(pos_weight))
    weights = [safe_pos_weight if sample.has_fg else 1.0 for sample in samples]
    return torch.tensor(weights, dtype=torch.double)


def main():
    parser = argparse.ArgumentParser("Fine-tune ViT for RIDER CT tumor segmentation (SEG-based)")
    parser.add_argument("--manifest-csv", type=str, default="project_understanding/rider_seg_pairs.csv")
    parser.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr-mult", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hu-min", type=int, default=-1000)
    parser.add_argument("--hu-max", type=int, default=400)
    parser.add_argument("--max-bg-per-case", type=int, default=16)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "directml", "cpu"],
        help="Training device backend. DirectML is currently disabled for stability and falls back to CPU.",
    )
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--pos-sample-weight", type=float, default=2.0)
    parser.add_argument("--disable-balanced-sampler", action="store_true", default=False)
    parser.add_argument("--outdir", type=str, default="project_understanding/checkpoints/rider_vit_seg")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    set_seed(args.seed)

    device, device_backend = select_training_device(args.device)
    use_cuda_amp = device_backend == "cuda"
    if args.device == "directml":
        print("DirectML disabled for stability. Falling back to CPU.")
    manifest_csv = resolve_path(args.manifest_csv, repo_root)
    outdir = resolve_path(args.outdir, repo_root)
    outdir.mkdir(parents=True, exist_ok=True)

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    pairs = load_pair_rows(manifest_csv)

    train_samples = build_slice_samples(
        pairs,
        split_name="train",
        pos_only=False,
        max_bg_per_case=args.max_bg_per_case,
        seed=args.seed,
    )
    val_samples = build_slice_samples(
        pairs,
        split_name="val",
        pos_only=False,
        max_bg_per_case=max(8, args.max_bg_per_case // 2),
        seed=args.seed + 1,
    )

    if not train_samples:
        raise RuntimeError("No train samples built from manifest (SEG pairing failed or empty split).")
    if not val_samples:
        raise RuntimeError("No val samples built from manifest (SEG pairing failed or empty split).")

    train_ds = RiderSegSliceDataset(
        train_samples,
        image_size=args.image_size,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        augment=True,
    )
    val_ds = RiderSegSliceDataset(
        val_samples,
        image_size=args.image_size,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        augment=False,
    )

    use_balanced_sampler = not args.disable_balanced_sampler
    if use_balanced_sampler:
        weights = build_balanced_weights(train_samples, pos_weight=args.pos_sample_weight)
        train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=use_cuda_amp,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=use_cuda_amp,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda_amp,
    )

    model = ViTSegHead(args.model_name).to(device)

    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]

    if device_backend == "directml":
        optimizer = torch.optim.SGD(
            [
                {"params": encoder_params, "lr": args.lr * args.encoder_lr_mult},
                {"params": decoder_params, "lr": args.lr},
            ],
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=False,
        )
        optimizer_name = "SGD"
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": args.lr * args.encoder_lr_mult},
                {"params": decoder_params, "lr": args.lr},
            ],
            weight_decay=args.weight_decay,
            foreach=False,
        )
        optimizer_name = "AdamW"
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    if not (0.0 <= args.bce_weight <= 1.0):
        raise ValueError("--bce-weight must be in [0, 1].")

    effective_bce_weight = float(args.bce_weight)
    if device_backend == "directml" and effective_bce_weight > 0.0:
        print(
            "DirectML stability mode: forcing BCE optimization weight to 0.0 "
            "to avoid unsupported log_sigmoid GPU fallback crashes."
        )
        effective_bce_weight = 0.0

    print(f"Pairs: {len(pairs)} | train slices: {len(train_samples)} | val slices: {len(val_samples)}")
    print(f"Device: {device} ({device_backend}) | model: {args.model_name}")
    print(
        f"Loss: total={effective_bce_weight:.2f}*BCE + {1.0 - effective_bce_weight:.2f}*DiceLoss | "
        f"balanced_sampler={use_balanced_sampler}"
    )
    print(f"Optimizer: {optimizer_name}")

    best_dice = -1.0
    start_epoch = 1

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

        if device_backend != "directml":
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            print("DirectML resume: skipping optimizer/scheduler state restore for stability.")
        if use_cuda_amp and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_dice = float(ckpt.get("best_dice", ckpt.get("val_metrics", {}).get("dice", -1.0)))
        print(f"Resume start epoch: {start_epoch} | best_dice={best_dice:.4f}")

    if start_epoch > args.epochs:
        print(
            f"Checkpoint already at epoch {start_epoch - 1}, which is >= --epochs ({args.epochs}). "
            "Nothing to train."
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()

        if epoch <= args.freeze_epochs:
            for p in model.encoder.parameters():
                p.requires_grad = False
        elif epoch == args.freeze_epochs + 1:
            for p in model.encoder.parameters():
                p.requires_grad = True

        train_total_loss = 0.0
        train_bce = 0.0
        train_dice = 0.0
        n_train = 0

        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            amp_context = torch.amp.autocast("cuda", enabled=True) if use_cuda_amp else nullcontext()
            with amp_context:
                logits = model(images)
                total_loss, bce_loss, _ = combined_segmentation_loss(
                    logits,
                    masks,
                    bce_weight=effective_bce_weight,
                )

            if use_cuda_amp:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            bs = images.size(0)
            train_total_loss += float(total_loss.item()) * bs
            if device_backend == "directml" or effective_bce_weight <= 0.0:
                train_bce += float(bce_metric_on_cpu(logits, masks).item()) * bs
            else:
                train_bce += float(bce_loss.item()) * bs
            train_dice += dice_score_from_logits(logits.detach(), masks) * bs
            n_train += bs

        scheduler.step()

        train_metrics = {
            "loss": train_total_loss / max(n_train, 1),
            "bce": train_bce / max(n_train, 1),
            "dice": train_dice / max(n_train, 1),
        }
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            use_inference_mode=(device_backend != "directml"),
            train_bce_weight=effective_bce_weight,
            device_backend=device_backend,
        )

        state = {
            "epoch": epoch,
            "model_name": args.model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_cuda_amp else None,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_dice": best_dice,
            "args": vars(args),
        }

        torch.save(state, outdir / "last.pt")
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(state, outdir / "best.pt")

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_bce={train_metrics['bce']:.4f} | train_dice={train_metrics['dice']:.4f} | "
            f"val_bce={val_metrics['bce']:.4f} | val_dice={val_metrics['dice']:.4f}"
        )

    print(f"Training complete. Best val Dice: {best_dice:.4f}")
    print(f"Saved checkpoints in: {outdir}")


if __name__ == "__main__":
    main()
