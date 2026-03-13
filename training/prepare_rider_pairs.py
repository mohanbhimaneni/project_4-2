import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple


def _resolve(path_value: str, repo_root: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p

    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (repo_root / p).resolve()


def _parse_int(value: str, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _norm_rel_path(rel: str) -> Path:
    rel = rel.replace(".\\", "").replace("\\", "/").strip("/")
    return Path(rel)


def _session_from_series_description(desc: str) -> str:
    d = (desc or "").upper()
    if "RETEST" in d:
        return "retest"
    if "TEST" in d:
        return "test"
    return "unknown"


def load_ct_rows(ct_metadata_csv: Path, ct_root: Path) -> Dict[str, List[dict]]:
    by_study: Dict[str, List[dict]] = {}

    with ct_metadata_csv.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("Modality") or "").strip().upper() != "CT":
                continue

            study_uid = (row.get("Study UID") or "").strip()
            if not study_uid:
                continue

            rel_loc = (row.get("File Location") or "").strip()
            if not rel_loc:
                continue

            folder = ct_root / _norm_rel_path(rel_loc)
            if not folder.exists():
                continue

            row_copy = dict(row)
            row_copy["_ct_folder"] = str(folder)
            row_copy["_num_images"] = _parse_int(row.get("Number of Images"), default=0)
            by_study.setdefault(study_uid, []).append(row_copy)

    return by_study


def load_seg_rows(mask_metadata_csv: Path, mask_root: Path) -> List[dict]:
    rows: List[dict] = []

    with mask_metadata_csv.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            modality = (row.get("Modality") or "").strip().upper()
            if modality != "SEG":
                continue

            rel_loc = (row.get("File Location") or "").strip()
            if not rel_loc:
                continue

            folder = mask_root / _norm_rel_path(rel_loc)
            if not folder.exists():
                continue

            seg_files = sorted(folder.glob("*.dcm"))
            if not seg_files:
                continue

            row_copy = dict(row)
            row_copy["_seg_file"] = str(seg_files[0])
            rows.append(row_copy)

    return rows


def assign_patient_level_splits(subject_ids: List[str], seed: int, val_ratio: float, test_ratio: float) -> Dict[str, str]:
    ids = sorted(set(subject_ids))
    rnd = random.Random(seed)
    rnd.shuffle(ids)

    n_total = len(ids)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))

    n_test = max(1 if n_total >= 3 and test_ratio > 0 else 0, min(n_test, n_total))
    n_val = max(1 if n_total >= 4 and val_ratio > 0 else 0, min(n_val, max(0, n_total - n_test)))

    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test:n_test + n_val])

    split_map: Dict[str, str] = {}
    for sid in ids:
        if sid in test_ids:
            split_map[sid] = "test"
        elif sid in val_ids:
            split_map[sid] = "val"
        else:
            split_map[sid] = "train"

    return split_map


def build_pairs(ct_by_study: Dict[str, List[dict]], seg_rows: List[dict]) -> Tuple[List[dict], int]:
    pairs: List[dict] = []
    missing_ct = 0

    for seg in seg_rows:
        study_uid = (seg.get("Study UID") or "").strip()
        subject_id = (seg.get("Subject ID") or "").strip()
        seg_series_uid = (seg.get("Series UID") or "").strip()
        seg_desc = (seg.get("Series Description") or "").strip()

        ct_candidates = ct_by_study.get(study_uid, [])
        if not ct_candidates:
            missing_ct += 1
            continue

        # Choose CT series with highest image count for stable volumetric pairing.
        ct_best = sorted(ct_candidates, key=lambda x: x.get("_num_images", 0), reverse=True)[0]

        pairs.append(
            {
                "subject_id": subject_id,
                "study_uid": study_uid,
                "session": _session_from_series_description(seg_desc),
                "ct_series_uid": (ct_best.get("Series UID") or "").strip(),
                "seg_series_uid": seg_series_uid,
                "ct_dir": ct_best["_ct_folder"],
                "seg_file": seg["_seg_file"],
            }
        )

    return pairs, missing_ct


def save_pairs_csv(pairs: List[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subject_id",
        "study_uid",
        "session",
        "split",
        "ct_series_uid",
        "seg_series_uid",
        "ct_dir",
        "seg_file",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in pairs:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser("Build RIDER CT+SEG pair manifest for segmentation training")
    parser.add_argument("--ct-metadata", type=str, default="dataset/RIDER_CT/metadata.csv")
    parser.add_argument("--mask-metadata", type=str, default="dataset/RIDER_masks/metadata.csv")
    parser.add_argument("--ct-root", type=str, default="dataset/RIDER_CT")
    parser.add_argument("--mask-root", type=str, default="dataset/RIDER_masks")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=str, default="project_understanding/rider_seg_pairs.csv")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    ct_metadata = _resolve(args.ct_metadata, repo_root)
    mask_metadata = _resolve(args.mask_metadata, repo_root)
    ct_root = _resolve(args.ct_root, repo_root)
    mask_root = _resolve(args.mask_root, repo_root)
    out_csv = _resolve(args.out_csv, repo_root)

    if not ct_metadata.exists():
        raise FileNotFoundError(f"CT metadata not found: {ct_metadata}")
    if not mask_metadata.exists():
        raise FileNotFoundError(f"Mask metadata not found: {mask_metadata}")
    if not ct_root.exists():
        raise FileNotFoundError(f"CT root not found: {ct_root}")
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask root not found: {mask_root}")

    ct_by_study = load_ct_rows(ct_metadata, ct_root)
    seg_rows = load_seg_rows(mask_metadata, mask_root)
    pairs, missing_ct = build_pairs(ct_by_study, seg_rows)

    if not pairs:
        raise RuntimeError("No CT+SEG pairs found. Check metadata paths and modality filters.")

    split_map = assign_patient_level_splits(
        subject_ids=[p["subject_id"] for p in pairs],
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    for p in pairs:
        p["split"] = split_map[p["subject_id"]]

    save_pairs_csv(pairs, out_csv)

    n_train = sum(1 for p in pairs if p["split"] == "train")
    n_val = sum(1 for p in pairs if p["split"] == "val")
    n_test = sum(1 for p in pairs if p["split"] == "test")
    n_subjects = len(set(p["subject_id"] for p in pairs))

    print(f"Saved: {out_csv}")
    print(f"Pairs: {len(pairs)} | subjects: {n_subjects}")
    print(f"Split counts -> train: {n_train}, val: {n_val}, test: {n_test}")
    if missing_ct > 0:
        print(f"Warning: {missing_ct} SEG rows had no matching CT study UID and were skipped.")


if __name__ == "__main__":
    main()
