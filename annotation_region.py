"""
annotation_region.py — Qupath 圈画 ROI 与 H-score 阈值模块 (IMC-HistoTools)

提供:
  * ``virtual_IHC_annotation`` : 读取 Qupath geojson 圈画文件 → 栅格化为 ROI mask
  * ``Hscore_threshold``       : 根据参考 ROI (Np / Lp / Mp / Hp) 计算各层
                                 像素中位值, 作为 pixel-wise H-score 分层阈值

对应 2610/261003_Pix_Hscore_histopathology.ipynb 中的 geojson → mask 逻辑,
并扩展为可直接复用的包函数。
"""

import json
from pathlib import Path

import numpy as np
import cv2


# ============================================================
# helper
# ============================================================

def _as_mask(roi, shape):
    """把「单个 mask 或 mask 列表」归一化为一个 bool ndarray (列表取并集)."""
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
# 1. Qupath geojson → ROI mask
# ============================================================

def virtual_IHC_annotation(image, geojson_path, region_order=None, name_key='name'):
    """
    读取 Qupath geojson 圈画文件, 栅格化为 ROI 区域的布尔 mask。

    仅取每个 polygon 的外环 (第一个坐标环), 用 ``cv2.fillPoly`` 填充;
    支持 Polygon 与 MultiPolygon。

    Parameters
    ----------
    image : 2D ndarray 或 (H, W) tuple
        用于确定栅格尺寸 H, W (通常传入对应的 virtual IHC / marker 图像)。
    geojson_path : str or Path
        Qupath 导出的 .geojson 文件路径。
    region_order : list of str or None
        返回结果的 region 顺序; None → 按 geojson 中出现顺序去重。
    name_key : str
        feature['properties'] 中用于区分区域名称的键 (默认 'name')。

    Returns
    -------
    dict
        {
          'polys' : {name: [xy ndarray (N,2), ...]},   # 每个区域的外环顶点 (x, y)
          'masks' : {name: bool ndarray (H, W)},       # 每个区域的栅格 mask
          'names' : [按顺序的 region 名],
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
        geom = feat['geometry']
        if geom['type'] == 'MultiPolygon':
            polys = geom['coordinates']
        else:
            polys = [geom['coordinates']]
        for poly in polys:
            # 外环 (x, y)
            region_polys.setdefault(name, []).append(
                np.asarray(poly[0], dtype=np.float64))

    if region_order is None:
        region_order = list(region_polys.keys())

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
# 2. 参考 ROI → H-score 分层阈值 (中位值)
# ============================================================

def Hscore_threshold(marker_image, Np=None, Lp=None, Mp=None, Hp=None,
                     tissue_mask=None, exclude_zero=False, stat='median'):
    """
    根据参考 ROI (Np / Lp / Mp / Hp) 统计像素强度, 作为返回的阈值。

    对每个参考 ROI 分别统计:
      - mean    : 平均值
      - median  : 中位值
      - geomean : 几何平均值 (仅正像素)
      - all     : mean / median / geomean 三者的平均

    Np / Lp / Mp / Hp 分别对应单个或多个设置 ROI 区域 (bool mask 或 mask 列表,
    列表取并集); 可传 ``tissue_mask`` 以仅统计落在组织内的像素。

    Parameters
    ----------
    marker_image : 2D ndarray
        marker 强度图像 (通常为背景扣除后的图像)。
    Np, Lp, Mp, Hp : bool ndarray 或 list of bool ndarray, 或 None
        负 / 低 / 中 / 高 表达的参考 ROI 区域。
    tissue_mask : bool ndarray or None
        组织 mask (True=组织); 提供时各 ROI 会与组织求交。
    exclude_zero : bool
        是否在统计前排除强度为 0 的像素 (默认 False)。
    stat : str, {'mean','median','geomean','all'}
        选择用于 ``thresholds`` 的统计量 (默认 'median')。

    Returns
    -------
    dict
        {
          'Np' : {'mean','median','geomean','all'} 或 None,
          'Lp' : {...}, 'Mp' : {...}, 'Hp' : {...},
          'thresholds' : (T1, T2, T3),   # 由 stat 决定, 对应 Lp/Mp/Hp
          'stat' : str,
        }
    """
    if stat not in ('mean', 'median', 'geomean', 'all'):
        raise ValueError(f"stat 必须是 'mean'/'median'/'geomean'/'all', 得到 {stat!r}")

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
