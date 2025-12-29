import numpy as np
from core.utils import getionimage, bisect_spectrum
from core.volcano_analytics import (
    build_global_spectrum_for_features,
    select_volcano_features,
    compute_volcano_statistics)
from pyimzml.ImzMLParser import ImzMLParser
try:
    from scipy.stats import (
        gaussian_kde, norm,
        mannwhitneyu,
        kruskal, f_oneway, levene, shapiro
    )
    _HAS_SCIPY_STATS = True
except Exception:
    _HAS_SCIPY_STATS = False
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
    import numpy as np
    from scipy.stats import mannwhitneyu
    import logging

    print(">>> task_roi_volcano STARTED")
    logger = logging.getLogger(__name__)

    empty_payload = {
        "mz": np.array([]),
        "logFC": np.array([]),
        "p_raw": np.array([]),
        "q": np.array([]),
        "reason": ""
    }

    try:
        # ---------- ROI CHECK ----------
        infoA = saved_rois.get(nameA)
        infoB = saved_rois.get(nameB)
        if infoA is None or infoB is None:
            empty_payload["reason"] = "roi_not_found"
            return empty_payload

        maskA, zA = infoA["mask"], int(infoA["z"])
        maskB, zB = infoB["mask"], int(infoB["z"])

        # ---------- PIXEL INDEX (DOWNSAMPLING) ----------
        idxA = np.where(maskA.ravel())[0]
        idxB = np.where(maskB.ravel())[0]

        if idxA.size > max_pixels:
            idxA = np.random.choice(idxA, max_pixels, replace=False)
        if idxB.size > max_pixels:
            idxB = np.random.choice(idxB, max_pixels, replace=False)

        # ---------- FEATURE SELECTION ----------
        mz_grid, inten = build_global_spectrum_for_features(
            parser,
            max_pixels=max_pixels
        )
        features = select_volcano_features(mz_grid, inten)

        if features is None or len(features) < 5:
            empty_payload["reason"] = "not_enough_features"
            return empty_payload

        # ---------- FEATURE LIMIT ----------
        MAX_FEATURES = 120
        if hasattr(parser, "large_dataset_mode") and parser.large_dataset_mode:
            MAX_FEATURES = 10
            logger.info(
                "Large dataset mode ENABLED → feature limit set to 10 (Volcano)"
            )

        features = features[:MAX_FEATURES]

        logfc = []
        pvals = []
        valid_mz = []

        # ---------- MAIN LOOP ----------
        for i, mzv in enumerate(features):
            print(f">>> VOLCANO LOOP i={i} mz={mzv:.4f}")

            img = getionimage(parser, mzv, tol, zA, cancel_cb=cancel_cb)
            if img is None:
                continue

            flat = img.ravel()

            vA = flat[idxA]
            vB = flat[idxB]

            # 🔑 SADECE NaN temizle (ZERO KALSIN)
            vA = vA[np.isfinite(vA)]
            vB = vB[np.isfinite(vB)]

            print(f">>> vA size={vA.size}, vB size={vB.size}")

            if vA.size < 3 or vB.size < 3:
                continue

            # ---------- ROBUST LOG2FC (median-based) ----------
            medA = np.median(vA)
            medB = np.median(vB)
            logfc.append(np.log2((medB + 1e-9) / (medA + 1e-9)))

            # ---------- STAT TEST ----------
            _, p = mannwhitneyu(vA, vB, alternative="two-sided")
            pvals.append(p)
            valid_mz.append(mzv)

            if progress_cb:
                progress_cb(int((i + 1) / len(features) * 100))

        # ---------- FINAL CHECK ----------
        if len(pvals) < 3:
            empty_payload["reason"] = "not_enough_valid_features"
            return empty_payload

        logfc = np.asarray(logfc)
        pvals = np.asarray(pvals)

        # ---------- BH-FDR ----------
        order = np.argsort(pvals)
        ranked = pvals[order]
        m = len(ranked)

        qvals = np.empty_like(ranked)
        prev = 1.0
        for i in reversed(range(m)):
            q = ranked[i] * m / (i + 1)
            q = min(q, prev)
            qvals[i] = q
            prev = q

        qvals_corrected = np.empty_like(qvals)
        qvals_corrected[order] = qvals

        print(f">>> task_roi_volcano RETURNING {len(valid_mz)} FEATURES")

        return {
            "mz": np.asarray(valid_mz),
            "logFC": logfc,
            "p_raw": pvals,
            "q": qvals_corrected
        }

    except Exception as e:
        empty_payload["reason"] = f"exception: {str(e)}"
        return empty_payload





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
