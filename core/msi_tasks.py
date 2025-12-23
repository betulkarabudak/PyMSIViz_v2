import numpy as np
from core.utils import getionimage, bisect_spectrum
from core.volcano_analytics import (
    build_global_spectrum_for_features,
    select_volcano_features,
    compute_volcano_statistics)
from pyimzml.ImzMLParser import ImzMLParser

def task_compute_ionimage( parser, mz, tol, z, reduce_func=sum, *, progress_cb=None, cancel_cb=None):
    tol = abs(float(tol))
    mz = float(mz)
    z = int(z)

    tol = abs(float(tol))
    mz = float(mz)
    z = int(z)

    max_y = parser.imzmldict["max count of pixels y"]
    max_x = parser.imzmldict["max count of pixels x"]
    im = np.zeros((max_y, max_x), dtype=float)

    coords = parser.coordinates
    total = len(coords)

    if total == 0:
        return {
            "image": im,
            "mz": mz,
            "tol": tol,
            "z": z,
            "cancelled": False
        }

    step = max(1, total // 100)

    for i, (x, y, z_) in enumerate(coords):
        if cancel_cb and cancel_cb():
            return {
                "image": im,
                "mz": mz,
                "tol": tol,
                "z": z,
                "cancelled": True
            }

        if z_ == z:
            mzs, ints = map(np.asarray, parser.getspectrum(i))
            min_i, max_i = bisect_spectrum(mzs, mz, tol)
            if max_i >= min_i:
                im[y - 1, x - 1] = reduce_func(ints[min_i:max_i + 1])

        if progress_cb and (i % step == 0):
            progress_cb(int((i / total) * 100))

    if progress_cb:
        progress_cb(100)

    return {
        "image": im,
        "mz": mz,
        "tol": tol,
        "z": z,
        "cancelled": False
    }

def task_ai_assist_top_peak(
    parser,
    *,
    max_pixels=1200,
    n_bins=6000
):
    """
    Global mean spectrum üzerinden en yüksek piki bulur.
    SADECE hesap yapar, UI YOK.
    """

    mz, inten = build_global_spectrum_for_features(
        parser,
        max_pixels=max_pixels,
        n_bins=n_bins
    )

    if mz is None or mz.size == 0:
        return None

    return float(mz[int(np.argmax(inten))])

def task_roi_compare_violin(
    parser,
    roi_mgr,
    selected,
    mz1, tol1,
    mz2, tol2,
    mz3, tol3,
    use_multi,
    *,
    max_pixels=None,
    progress_cb=None,
    cancel_cb=None
):
    data = []
    labels = []

    infoA = roi_mgr.get_roi(selected[0])
    infoB = roi_mgr.get_roi(selected[1])

    if infoA is None or infoB is None:
        return {"data": [], "labels": []}

    maskA, zA = infoA["mask"], int(infoA["z"])
    maskB, zB = infoB["mask"], int(infoB["z"])

    if zA != zB:
        return {"data": [], "labels": []}

    z = zA

    mz_list = [(mz1, tol1)]
    if use_multi:
        if mz2 is not None and tol2 is not None:
            mz_list.append((mz2, tol2))
        if mz3 is not None and tol3 is not None:
            mz_list.append((mz3, tol3))

    for i, (name, mask) in enumerate(
        [(selected[0], maskA), (selected[1], maskB)]
    ):
        if cancel_cb and cancel_cb():
            return {"data": [], "labels": []}

        vals_all = []

        for mz, tol in mz_list:
            if cancel_cb and cancel_cb():
                return {"data": [], "labels": []}

            img = getionimage(parser, float(mz), float(tol), z)

            if img is None or img.shape != mask.shape:
                continue

            vals = img[mask].astype(float)
            vals = vals[np.isfinite(vals)]

            if vals.size > 0:
                vals_all.append(vals)

        if vals_all:
            merged = np.concatenate(vals_all)
            data.append(merged)
            labels.append(name)

        if progress_cb:
            progress_cb(int(((i + 1) / 2) * 100))

    if len(data) != 2:
        return {"data": [], "labels": []}

    return {
        "data": data,
        "labels": labels
    }

def task_roi_volcano(
    parser,
    saved_rois,
    nameA,
    nameB,
    tol,
    max_pixels=1500,
    alpha=0.05,
    *,
    progress_cb=None,
    cancel_cb=None
):
    empty_payload = {
        "log2fc": np.array([]),
        "y": np.array([]),
        "rejected": np.array([]),
        "reason": ""
    }

    try:
        infoA = saved_rois.get(nameA)
        infoB = saved_rois.get(nameB)

        if infoA is None or infoB is None:
            empty_payload["reason"] = "roi_not_found"
            return empty_payload

        maskA, zA = infoA["mask"], int(infoA["z"])
        maskB, zB = infoB["mask"], int(infoB["z"])

        mz, inten = build_global_spectrum_for_features(
            parser,
            max_pixels=max_pixels
        )

        features = select_volcano_features(mz, inten)

        # 🔬 Bilimsel alt sınır (çok kritik)
        if features is None or len(features) < 5:
            empty_payload["reason"] = "not_enough_features"
            return empty_payload

        values_A = []
        values_B = []

        n_feat = len(features)

        for i, mzv in enumerate(features):
            if cancel_cb and cancel_cb():
                empty_payload["reason"] = "cancelled"
                return empty_payload

            imgA = getionimage(
                parser, mzv, tol, zA,
                cancel_cb=cancel_cb,
                progress_cb=None
            )
            imgB = getionimage(
                parser, mzv, tol, zB,
                cancel_cb=cancel_cb,
                progress_cb=None
            )

            if imgA is None or imgB is None:
                continue

            vA = imgA[maskA].astype(float)
            vB = imgB[maskB].astype(float)

            if vA.size < 3 or vB.size < 3:
                continue

            values_A.append(vA)
            values_B.append(vB)

            if progress_cb:
                progress_cb(int((i + 1) / max(1, n_feat) * 100))

        if len(values_A) < 5 or len(values_B) < 5:
            empty_payload["reason"] = "not_enough_valid_features"
            return empty_payload

        payload = compute_volcano_statistics(
            feature_mz=features,
            values_A=values_A,
            values_B=values_B,
            alpha=alpha)

        if payload is None or payload.get("log2fc", np.array([])).size == 0:
            empty_payload["reason"] = "no_significant_features"
            return empty_payload

        return payload

    except Exception as e:
        empty_payload["reason"] = f"exception: {str(e)}"
        return empty_payload

from pyimzml.ImzMLParser import ImzMLParser

def task_load_imzml_safe(filename):
    """
    SAFE loader for large imzML datasets.
    - No full spectrum traversal
    - No heavy RAM usage
    - Estimates m/z range from representative spectra
    """
    parser = ImzMLParser(filename)

    mz_min = None
    mz_max = None

    coords = parser.coordinates
    if coords:
        for i in range(min(10, len(coords))):
            mzs, _ = parser.getspectrum(i)
            if mzs is not None and len(mzs) > 0:
                mz_min = float(mzs[0])
                mz_max = float(mzs[-1])
                break

    if mz_min is None or mz_max is None:
        mz_min, mz_max = 0.0, 0.0

    info = {
        "mz_min": mz_min,
        "mz_max": mz_max
    }

    return parser, info
