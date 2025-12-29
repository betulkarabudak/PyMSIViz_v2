import numpy as np
import warnings

def benjamini_hochberg(pvals):

    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size

    if m == 0:
        return np.array([])

    order = np.argsort(pvals)
    ranked = pvals[order]

    # BH adjustment
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)

    p_adj = np.empty_like(adj)
    p_adj[order] = adj

    return p_adj

def peak_pick_features(mz, inten, top_k=300, min_mz_dist=0.15, snr=3.0):
    mz = np.asarray(mz, dtype=float)
    inten = np.asarray(inten, dtype=float)
    if mz.size < 10:
        return np.array([], dtype=float)

    med = np.median(inten)
    mad = np.median(np.abs(inten - med)) + 1e-12
    robust_sigma = 1.4826 * mad

    sig = inten - med
    sig[sig < 0] = 0.0

    if sig.size < 3:
        return np.array([], dtype=float)

    left = sig[:-2]
    mid = sig[1:-1]
    right = sig[2:]
    is_peak = (mid > left) & (mid >= right) & (mid > (snr * robust_sigma))

    peak_idx = np.where(is_peak)[0] + 1
    if peak_idx.size == 0:
        return np.array([], dtype=float)

    order = peak_idx[np.argsort(sig[peak_idx])[::-1]]
    picked = []

    for idx in order:
        mz_i = mz[idx]
        if all(abs(mz_i - mz[j]) >= min_mz_dist for j in picked):
            picked.append(idx)
        if len(picked) >= top_k:
            break
    picked = np.array(picked, dtype=int)
    picked = picked[np.argsort(mz[picked])]
    return mz[picked]
def compute_global_spectrum_binned(parser,
    max_pixels=1200,
    n_bins=6000,
    stat="mean"
    ):
    return compute_global_spectrum_binned(
        parser,
        max_pixels=max_pixels,
        n_bins=n_bins,
        stat=stat
            )    
def compute_mean_spectrum(parser, max_pixels=800):
    n = len(parser.coordinates)
    if n == 0:
        raise ValueError("No pixels found.")

    if n <= max_pixels:
        idxs = np.arange(n)
    else:
        idxs = np.random.choice(n, size=max_pixels, replace=False)

    mzs_list, int_list = [], []
    for idx in idxs:
        try:
            mzs, ints = parser.getspectrum(int(idx))
            if len(mzs) > 0:
                mzs_list.append(np.asarray(mzs))
                int_list.append(np.asarray(ints))
        except Exception:
            continue

    if not mzs_list:
        raise ValueError("No valid spectra found in sample.")

    lengths = [len(m) for m in mzs_list]
    if len(set(lengths)) == 1:
        df_mz = np.mean(np.vstack(mzs_list), axis=0)
        df_int = np.mean(np.vstack(int_list), axis=0)
        return df_mz, df_int

    all_mz = np.concatenate(mzs_list)
    all_int = np.concatenate(int_list)
    bins = np.linspace(all_mz.min(), all_mz.max(), 6000)
    dig = np.digitize(all_mz, bins)

    df_mz, df_int = [], []
    for i in range(1, len(bins)):
        mask = (dig == i)
        if mask.any():
            df_mz.append(bins[i])
            df_int.append(all_int[mask].mean())

    return np.array(df_mz), np.array(df_int)


def compute_mean_spectrum_legacy(parser, max_pixels=800):
    warnings.warn(
        "compute_mean_spectrum_legacy is deprecated. "
        "Use compute_global_spectrum_binned instead.",
        DeprecationWarning
    )
    return compute_mean_spectrum(parser, max_pixels)

def compute_global_spectrum_binned(
    parser,
    max_pixels=1200,
    n_bins=6000,
    stat="mean"
):
    n = len(parser.coordinates)
    if n == 0:
        raise ValueError("No pixels found.")

    if n <= max_pixels:
        idxs = np.arange(n, dtype=int)
    else:
        idxs = np.random.choice(n, size=max_pixels, replace=False).astype(int)

    mz_min = None
    mz_max = None
    spectra = []

    for idx in idxs:
        mzs, ints = parser.getspectrum(int(idx))
        if mzs is None or len(mzs) == 0:
            continue
        mzs = np.asarray(mzs, dtype=float)
        ints = np.asarray(ints, dtype=float)

        spectra.append((mzs, ints))
        mz_min = mzs[0] if mz_min is None else min(mz_min, mzs[0])
        mz_max = mzs[-1] if mz_max is None else max(mz_max, mzs[-1])

    if not spectra or mz_max <= mz_min:
        raise ValueError("No valid spectra found to build global grid.")

    edges = np.linspace(mz_min, mz_max, int(n_bins) + 1)
    grid = 0.5 * (edges[:-1] + edges[1:])

    M = np.zeros((len(spectra), int(n_bins)), dtype=np.float32)

    for r, (mzs, ints) in enumerate(spectra):
        bi = np.searchsorted(edges, mzs, side="right") - 1
        valid = (bi >= 0) & (bi < n_bins)
        np.add.at(M[r], bi[valid], ints[valid])

    stat = (stat or "mean").lower()
    agg = np.median(M, axis=0) if stat == "median" else np.mean(M, axis=0)

    return grid.astype(float), agg.astype(float)






