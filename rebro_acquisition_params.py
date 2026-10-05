#!/usr/bin/env python3
"""Per-patient MRI acquisition parameters of the 12 in-study internal (KBC Rebro)
patients, read from the DICOM headers of the exact T2 sagittal series used in
the study. Feeds the paper's acquisition table (Table tab:acq).

The series used per patient is taken from dataset/dicom_converted/
conversion_summary.csv (written by dicom_to_slices.py), so the parameters are
tied to the converted series by SeriesInstanceUID rather than guessed from the
series description.

Handles both classic single-frame DICOM and Enhanced MR multi-frame DICOM
(newer Siemens XA exports, e.g. reb013/reb014), where TR/TE/slice thickness/
pixel spacing are stored inside the Shared/Per-frame Functional Groups.

Output:
    comparison_results/rebro_acquisition_params.csv   one row per patient
    stdout                                            table + value ranges

Usage:
    python rebro_acquisition_params.py [--workers 20]
"""
import argparse
import csv
import os
from collections import Counter

import pydicom
from joblib import Parallel, delayed
from tqdm import tqdm

DICOM_ROOT = "dataset/dicom/kbc_rebro/FERIT"
LABELS_CSV = "dataset/dicom/kbc_rebro/clinical_labels.csv"
SUMMARY_CSV = "dataset/dicom_converted/conversion_summary.csv"
OUT_CSV = "comparison_results/rebro_acquisition_params.csv"

# f-AnoGAN leave-healthy-out roster (5 train + 7 test = the 12 in-study patients)
TRAIN = ["reb003", "reb006", "reb009", "reb013", "reb014"]
TEST = ["reb001", "reb004", "reb005", "reb007", "reb008", "reb010", "reb011"]

ENHANCED_MR = "1.2.840.10008.5.1.4.1.1.4.1"


def _search(item, keyword, path=""):
    """Depth-first search for `keyword` inside a dataset item and its sequences.
    Returns (value, path) or (None, None)."""
    if keyword in item and item.get(keyword) not in (None, ""):
        return item.get(keyword), f"{path}{keyword}"
    for elem in item:
        if elem.VR == "SQ" and elem.value:
            for sub in elem.value:
                hit, p = _search(sub, keyword, f"{path}{elem.keyword}>")
                if hit is not None:
                    return hit, p
    return None, None


def find_tag_src(ds, *keywords):
    """First non-empty value of any keyword and where it was found: top level
    first, then the Shared and first Per-frame Functional Groups (Enhanced MR)."""
    for kw in keywords:
        v = ds.get(kw)
        if v not in (None, ""):
            return v, kw
    for fg in ("SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence"):
        seq = ds.get(fg)
        if not seq:
            continue
        for kw in keywords:
            hit, p = _search(seq[0], kw, f"{fg}[0]>")
            if hit is not None:
                return hit, p
    return None, ""


def find_tag(ds, *keywords):
    return find_tag_src(ds, *keywords)[0]


def per_frame_spread(ds, *keywords):
    """Enhanced MR only: distinct values of a parameter across all frames."""
    vals = set()
    for item in ds.get("PerFrameFunctionalGroupsSequence") or []:
        for kw in keywords:
            hit, _ = _search(item, kw)
            if hit is not None:
                vals.add(fmt(hit))
                break
    return vals


def acquired_matrix(ds):
    """Acquired (pre-reconstruction) matrix as 'frequency x phase'."""
    am = ds.get("AcquisitionMatrix")  # classic: [freq rows, freq cols, phase rows, phase cols]
    if am:
        vals = [int(v) for v in am]
        freq = vals[0] or vals[1]
        phase = vals[2] or vals[3]
        return f"{freq}x{phase}"
    freq = find_tag(ds, "MRAcquisitionFrequencyEncodingSteps")  # Enhanced MR
    phase = find_tag(ds, "MRAcquisitionPhaseEncodingStepsInPlane")
    if freq and phase:
        return f"{int(freq)}x{int(phase)}"
    return ""


def fmt(v):
    if v is None or v == "":
        return ""
    if isinstance(v, (list, tuple, pydicom.multival.MultiValue)):
        return "/".join(fmt(x) for x in v)
    try:
        f = float(v)
        return f"{f:g}"
    except (TypeError, ValueError):
        return str(v)


def read_patient(reb, folder, series_uid, series_desc):
    root = os.path.join(DICOM_ROOT, folder)
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs)
    matched = []
    for fp in files:
        try:
            h = pydicom.dcmread(fp, stop_before_pixels=True, force=True,
                                specific_tags=["SeriesInstanceUID"])
        except Exception:
            continue
        if str(h.get("SeriesInstanceUID", "")) == series_uid:
            matched.append(fp)
    row = {"patient": reb, "paper_id": "I" + reb[-2:],
           "role": "train" if reb in TRAIN else "test", "folder": folder,
           "series_description": series_desc, "n_files": len(matched)}
    if not matched:
        row["notes"] = "series UID not found on disk"
        return row

    per_file = []
    for fp in matched:
        ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
        enhanced = str(ds.get("SOPClassUID", "")) == ENHANCED_MR
        if enhanced:  # where each value lives + whether it is constant across frames
            src = {"tr_source": find_tag_src(ds, "RepetitionTime")[1],
                   "te_source": find_tag_src(ds, "EchoTime", "EffectiveEchoTime")[1],
                   "thk_source": find_tag_src(ds, "SliceThickness")[1],
                   "px_source": find_tag_src(ds, "PixelSpacing")[1]}
            spread = []
            for label, kws in [("TR", ("RepetitionTime",)), ("TE", ("EffectiveEchoTime",)),
                               ("thickness", ("SliceThickness",)), ("pixel spacing", ("PixelSpacing",))]:
                v = per_frame_spread(ds, *kws)
                if len(v) > 1:
                    spread.append(f"{label} varies across frames: {sorted(v)}")
            src["frame_check"] = "; ".join(spread) or "constant across all frames (or shared)"
        else:
            src = {"tr_source": "RepetitionTime", "te_source": "EchoTime",
                   "thk_source": "SliceThickness", "px_source": "PixelSpacing",
                   "frame_check": ""}
        src["acq_matrix_raw"] = fmt(ds.get("AcquisitionMatrix")) or (
            f"FreqSteps={fmt(find_tag(ds, 'MRAcquisitionFrequencyEncodingSteps'))}, "
            f"PhaseSteps={fmt(find_tag(ds, 'MRAcquisitionPhaseEncodingStepsInPlane'))}")
        per_file.append({**src,
            "format": "enhanced" if str(ds.get("SOPClassUID", "")) == ENHANCED_MR else "classic",
            "manufacturer": fmt(ds.get("Manufacturer")),
            "model": fmt(ds.get("ManufacturerModelName")),
            "field_T": fmt(find_tag(ds, "MagneticFieldStrength")),
            "sequence_name": fmt(find_tag(ds, "SequenceName", "PulseSequenceName")),
            # SE = spin echo (classic ScanningSequence); SPIN = spin echo (Enhanced MR EchoPulseSequence)
            "scanning_sequence": fmt(find_tag(ds, "ScanningSequence", "EchoPulseSequence")),
            "sequence_variant": fmt(find_tag(ds, "SequenceVariant")),
            "echo_train_length": fmt(find_tag(ds, "EchoTrainLength")),
            "tr_ms": fmt(find_tag(ds, "RepetitionTime")),
            "te_ms": fmt(find_tag(ds, "EchoTime", "EffectiveEchoTime")),
            "slice_thickness_mm": fmt(find_tag(ds, "SliceThickness")),
            "spacing_between_slices_mm": fmt(find_tag(ds, "SpacingBetweenSlices")),
            "pixel_spacing_mm": fmt(find_tag(ds, "PixelSpacing")),
            "stored_matrix": f"{ds.get('Rows', '')}x{ds.get('Columns', '')}",
            "acquired_matrix": acquired_matrix(ds),
            "n_frames": fmt(ds.get("NumberOfFrames", 1)),
        })
    # report the per-series value; flag any field that differs across files
    notes = []
    for key in per_file[0]:
        vals = Counter(p[key] for p in per_file)
        row[key] = vals.most_common(1)[0][0]
        if len(vals) > 1:
            notes.append(f"{key} varies: {dict(vals)}")
    row["notes"] = "; ".join(notes)
    return row


def num_range(rows, key, label=None):
    vals = []
    for r in rows:
        try:
            vals.append(float(str(r.get(key, "")).split("/")[0]))
        except ValueError:
            pass
    missing = [r["patient"] for r in rows if r.get(key, "") == ""]
    rng = f"{min(vals):g}-{max(vals):g}" if vals else "n/a"
    miss = f"  missing: {missing}" if missing else ""
    print(f"  {label or key:28s} {rng:14s} (n={len(vals)}/{len(rows)}){miss}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    folder_of = {r["reb_id"]: r["folder_name"]
                 for r in csv.DictReader(open(LABELS_CSV, encoding="utf-8")) if r["reb_id"]}
    study = TRAIN + TEST
    series = {}
    for r in csv.DictReader(open(SUMMARY_CSV, encoding="utf-8")):
        if r["patient_id"] in study and "sag" in r["series_description"].lower():
            series.setdefault(r["patient_id"], []).append(r)
    jobs = []
    for reb in sorted(study):
        cands = series.get(reb, [])
        if len(cands) != 1:
            raise SystemExit(f"{reb}: expected exactly one T2 sagittal series, found {len(cands)}")
        jobs.append((reb, folder_of[reb], cands[0]["series_uid"], cands[0]["series_description"]))

    results = Parallel(n_jobs=args.workers, return_as="generator")(
        delayed(read_patient)(*j) for j in jobs)
    rows = sorted(tqdm(results, total=len(jobs), desc="patients"), key=lambda r: r["patient"])

    cols = ["patient", "paper_id", "role", "folder", "series_description", "format",
            "manufacturer", "model", "field_T", "sequence_name", "scanning_sequence",
            "sequence_variant", "echo_train_length",
            "tr_ms", "te_ms", "slice_thickness_mm", "spacing_between_slices_mm",
            "pixel_spacing_mm", "stored_matrix", "acquired_matrix", "n_files", "n_frames", "notes",
            "acq_matrix_raw", "tr_source", "te_source", "thk_source", "px_source", "frame_check"]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'pid':4s} {'series':24s} {'fmt':8s} {'model':22s} {'T':>3s} {'seq':12s} {'ETL':>4s} "
          f"{'TR':>6s} {'TE':>5s} {'thk':>4s} {'sp':>5s} {'px':>9s} {'stored':>8s} {'acq':>8s}")
    for r in rows:
        print(f"{r['paper_id']:4s} {r['series_description'][:24]:24s} {r.get('format', ''):8s} "
              f"{r.get('model', '')[:22]:22s} {r.get('field_T', ''):>3s} {r.get('sequence_name', '')[:12]:12s} "
              f"{r.get('echo_train_length', ''):>4s} {r.get('tr_ms', ''):>6s} {r.get('te_ms', ''):>5s} "
              f"{r.get('slice_thickness_mm', ''):>4s} {r.get('spacing_between_slices_mm', ''):>5s} "
              f"{r.get('pixel_spacing_mm', '')[:9]:>9s} {r.get('stored_matrix', ''):>8s} {r.get('acquired_matrix', ''):>8s}")
        if r.get("notes"):
            print(f"     note: {r['notes']}")

    print("\n=== ranges across the 12 in-study patients ===")
    print(f"  {'scanner':28s} {dict(Counter(r.get('model', '') for r in rows))}")
    print(f"  {'field (T)':28s} {dict(Counter(r.get('field_T', '') for r in rows))}")
    for key, label in [("tr_ms", "TR (ms), all"), ("te_ms", "TE (ms)"),
                       ("slice_thickness_mm", "slice thickness (mm)"),
                       ("spacing_between_slices_mm", "slice spacing (mm)"),
                       ("pixel_spacing_mm", "in-plane resolution (mm)")]:
        num_range(rows, key, label)
    for kind in ("haste", "tse"):
        sub = [r for r in rows if kind in r["series_description"].lower()
               and not (kind == "tse" and "haste" in r["series_description"].lower())]
        num_range(sub, "tr_ms", f"TR (ms), {kind.upper()} only")
    print(f"  {'acquired matrix':28s} {sorted(set(r.get('acquired_matrix', '') for r in rows))}")
    print(f"  {'stored matrix':28s} {sorted(set(r.get('stored_matrix', '') for r in rows))}")

    print("\n=== where the values come from ===")
    for r in rows:
        print(f"  {r['paper_id']}: {r.get('format', '')}; AcquisitionMatrix raw = {r.get('acq_matrix_raw', '')}; "
              f"scanning sequence = {r.get('scanning_sequence', '')} ({r.get('sequence_variant', '')})")
        if r.get("format") == "enhanced":
            for k in ("tr_source", "te_source", "thk_source", "px_source"):
                print(f"       {k[:-7]:>4s} <- {r.get(k, '')}")
            print(f"       frames: {r.get('frame_check', '')}")
    print(f"\nsaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
