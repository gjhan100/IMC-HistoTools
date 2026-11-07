"""
annotation_region.py — Qupath ROI annotation and H-score thresholding (IMC-HistoTools)

Provides:
  * ``virtual_IHC_annotation`` : read a Qupath geojson annotation file -> rasterize into ROI masks
  * ``Hscore_threshold``       : compute per-level pixel statistics (Np / Lp / Mp / Hp) from
                                 reference ROIs and use them as pixel-wise H-score thresholds

Corresponds to the geojson -> mask logic in 2610/261003_Pix_Hscore_histopathology.ipynb,
extended into a directly reusable package function.
"""

import json
from pathlib import Path

import numpy as np
import cv2


# ============================================================
# helper
# ============================================================

def _as_mask(roi, shape):
    """Normalize a single mask or a list of masks into one bool ndarray (union for lists)."""
    if roi is None:
        return None
    if isinstance(roi, (list, tuple)):
        m = np.zeros(shape, dtype=bool)
        for r in roi:
            if r is None:
                continue
            m |= np.asarray(r, dtype=bool)
        return m
    return np.asarray(roi, dtype=bool)


# ============================================================
# 1. Qupath geojson -> ROI mask
# ============================================================

DEFAULT_REGIONS = ('Low', 'Medium', 'High')


def virtual_IHC_annotation(image, geojson_path, region_order=None, name_key='name',
                           regions=DEFAULT_REGIONS):
    """
    Read a Qupath geojson annotation file and rasterize the ROIs into boolean masks.

    Only the exterior ring of each polygon (the first coordinate ring) is used and
    filled with ``cv2.fillPoly``; both Polygon and MultiPolygon are supported.

    Parameters
    ----------
    image : 2D ndarray or (H, W) tuple
        Used to determine the raster size H, W (usually the corresponding
        virtual IHC / marker image).
    geojson_path : str or Path
        Path to the Qupath-exported .geojson file.
    region_order : list of str or None
        Order of regions in the returned result; None -> follow the ``regions`` /
        geojson order (deduplicated).
    name_key : str
        Key in feature['properties'] used to distinguish region names (default 'name').
    regions : sequence of str or None
        Only read regions with these names (default Low/Medium/High); None -> read all.

    Returns
    -------
    dict
        {
          'polys' : {name: [xy ndarray (N,2), ...]},   # exterior ring vertices (x, y) per region
          'masks' : {name: bool ndarray (H, W)},       # rasterized mask per region
          'names' : [region names in order],
        }
    """
    image = np.asarray(image)
    if image.ndim == 2:
        H, W = image.shape
    else:
        H, W = int(image[0]), int(image[1])

    with open(geojson_path) as f:
        gj = json.load(f)

    region_polys = {}              # name -> [exterior xy array, ...]
    for feat in gj['features']:
        name = feat['properties'].get(name_key, 'unknown')
        if regions is not None and name not in regions:
            continue
        geom = feat['geometry']
        if geom['type'] == 'MultiPolygon':
            polys = geom['coordinates']
        else:
            polys = [geom['coordinates']]
        for poly in polys:
            # exterior ring (x, y)
            region_polys.setdefault(name, []).append(
                np.asarray(poly[0], dtype=np.float64))

    if region_order is None:
        region_order = ([r for r in regions if r in region_polys]
                        if regions is not None else list(region_polys.keys()))
    else:
        region_order = [r for r in region_order if r in region_polys]

    region_masks = {}
    for name in region_order:
        m = np.zeros((H, W), np.uint8)
        for xy in region_polys.get(name, []):
            pts = np.round(xy).astype(np.int32).reshape(-1, 1, 2)   # cv2: (x, y)
            cv2.fillPoly(m, [pts], 1)
        region_masks[name] = m.astype(bool)

    return {
        'polys': {n: region_polys.get(n, []) for n in region_order},
        'masks': region_masks,
        'names': region_order,
    }


# ============================================================
# 2. reference ROI -> H-score level thresholds (median)
# ============================================================

def Hscore_threshold(marker_image, Np=None, Lp=None, Mp=None, Hp=None,
                     tissue_mask=None, exclude_zero=False, stat='median'):
    """
    Compute per-ROI intensity statistics (Np / Lp / Mp / Hp) for use as thresholds.

    For each reference ROI, the following statistics are computed:
      - mean    : arithmetic mean
      - median  : median
      - geomean : geometric mean (positive pixels only)
      - all     : mean of mean / median / geomean

    Np / Lp / Mp / Hp correspond to one or more reference ROI regions (a bool mask
    or a list of masks; lists are unioned); ``tissue_mask`` can be supplied to count
    only pixels inside the tissue.

    Parameters
    ----------
    marker_image : 2D ndarray
        Marker intensity image (usually background-subtracted).
    Np, Lp, Mp, Hp : bool ndarray or list of bool ndarray, or None
        Reference ROI regions for negative / low / medium / high expression.
    tissue_mask : bool ndarray or None
        Tissue mask (True = tissue); when provided, each ROI is intersected with it.
    exclude_zero : bool
        Whether to exclude zero-intensity pixels before computing statistics (default False).
    stat : str, {'mean','median','geomean','all'}
        Statistic used for ``thresholds`` (default 'median').

    Returns
    -------
    dict
        {
          'Np' : {'mean','median','geomean','all'} or None,
          'Lp' : {...}, 'Mp' : {...}, 'Hp' : {...},
          'thresholds' : (T1, T2, T3),   # determined by stat, corresponding to Lp/Mp/Hp
          'stat' : str,
        }
    """
    if stat not in ('mean', 'median', 'geomean', 'all'):
        raise ValueError(f"stat must be 'mean'/'median'/'geomean'/'all', got {stat!r}")

    img = np.asarray(marker_image, dtype=np.float32)
    shape = img.shape

    def _stats(roi):
        m = _as_mask(roi, shape)
        if m is None:
            return None
        if tissue_mask is not None:
            m = m & np.asarray(tissue_mask, dtype=bool)
        vals = img[m].astype(np.float64)
        if exclude_zero:
            vals = vals[vals > 0]
        if vals.size == 0:
            return None
        mean = float(np.mean(vals))
        median = float(np.median(vals))
        pos = vals[vals > 0]
        geomean = float(np.exp(np.mean(np.log(pos)))) if pos.size else 0.0
        all_ = (mean + median + geomean) / 3.0
        return {'mean': mean, 'median': median, 'geomean': geomean, 'all': all_}

    stats = {'Np': _stats(Np), 'Lp': _stats(Lp),
             'Mp': _stats(Mp), 'Hp': _stats(Hp)}

    thresholds = tuple(stats[k][stat] if stats[k] is not None else None
                       for k in ('Lp', 'Mp', 'Hp'))

    result = dict(stats)
    result['thresholds'] = thresholds
    result['stat'] = stat
    return result
