"""
pix_Hscore.py — pixel-wise H-score 分类与可视化模块 (IMC-HistoTools)

核心函数 ``pix_Hscore_map``:
  * 输入 marker 图像与参考 ROI (Np / Lp / Mp / Hp), 调取 ``Hscore_threshold``
    得到分层阈值
  * 对每个像素分类为 0/1/2/3 (NP / LP / MP / HP)
  * 标准返回: 逐像素分类 CSV 文件 + 分类着色图 (colorbar 在图外右侧)
  * 若 ``tissue_annotation`` 非空, 额外返回各区域 pix H-score 柱状图 + 比例/面积表

对应 2610/261001_CRC_Pix_Hscore.ipynb 与
2610/261003_Pix_Hscore_histopathology.ipynb 的核心逻辑。
"""

import os
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgba
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
import tifffile

from .annotation_region import Hscore_threshold, _as_mask


# ============================================================
# 常量
# ============================================================
SCORE_COLORS = ['#FFFFFF', '#2166AC', '#67AB49', '#FDBF2C', '#D73027']
SCORE_LABELS = ['0 (Neg)', '1 (Low)', '2 (Mod)', '3 (High)']
CATEGORY_NAMES = {0: 'NP', 1: 'LP', 2: 'MP', 3: 'HP'}
_CAT_ARR = np.array(['NP', 'LP', 'MP', 'HP'])


# ============================================================
# helper
# ============================================================

def _load(img):
    """ndarray 原样返回; str/Path 视为 tiff 路径读取."""
    if isinstance(img, (str, Path)):
        return np.asarray(tifffile.imread(str(img))).astype(np.float32)
    return np.asarray(img)


def _classify(marker_image, thresholds, blur_sigma=None, blur_kernel=None):
    """阈值分层 (0/1/2/3)。可选高斯模糊。

    thresholds = (T1, T2, T3) = (Lp, Mp, Hp):
      0  if data <  T1
      1  if T1 <= data <  T2
      2  if T2 <= data <  T3
      3  if data >= T3
    """
    data_f = np.asarray(marker_image, dtype=np.float32)
    if blur_kernel is None or blur_sigma in (None, 0):
        data_blur = data_f
    else:
        if isinstance(blur_kernel, int):
            blur_kernel = (blur_kernel, blur_kernel)
        data_blur = cv2.GaussianBlur(data_f, blur_kernel, sigmaX=blur_sigma)
    T1, T2, T3 = thresholds  # (Lp, Mp, Hp)
    scores_2d = np.select(
        [data_blur >= T3, data_blur >= T2, data_blur >= T1],
        [3, 2, 1], default=0,
    ).astype(np.uint8)
    return scores_2d


def _hscore_from_scores(scores_2d, sel, thresholds=(0, 0, 0)):
    """在布尔子集 sel 上计算 pix H-Score 与各层计数/比例。"""
    sc = scores_2d[sel]
    total = int(sc.size)
    counts = {k: int((sc == k).sum()) for k in (0, 1, 2, 3)}
    NP, LP, MP, HP = (counts[k] for k in (0, 1, 2, 3))
    pix_h = 100.0 * (3 * HP + 2 * MP + LP) / total if total > 0 else 0.0
    percents = {k: (100.0 * v / total if total > 0 else 0.0)
                for k, v in counts.items()}
    return {'pix_h': pix_h, 'counts': counts, 'percents': percents,
            'total': total, 'thresholds': thresholds}


# ============================================================
# 主函数
# ============================================================

def pix_Hscore_map(marker_image,
                   Hscore_threshold_annotation=None,
                   tissue_annotation=None,
                   tissue_mask=None,
                   thresholds=None,
                   stat='median',
                   delta=None,
                   kernel=None,
                   # --- 输出 ---
                   save_dir='.',
                   base_name='pix_Hscore',
                   # --- 分类着色图 (map) 样式 ---
                   map_figsize=(12, 9),
                   map_title=None,
                   map_cmap=SCORE_COLORS,
                   map_show=True,
                   map_colorbar=True,
                   map_colorbar_label='Score',
                   map_colorbar_labels=SCORE_LABELS,
                   map_tick_step=500,
                   map_dpi=150,
                   map_save=True,
                   xlabel='5μm / pixel',
                   ylabel='5μm / pixel',
                   axis_label_fontsize=34,
                   tick_fontsize=24,
                   # --- 柱状图 + 表 样式 ---
                   bar_order=None,
                   region_colors=None,
                   pixel_area_um2=25,
                   bar_figsize=(6, 6.5),
                   bar_show=True,
                   bar_save=True,
                   bar_dpi=150,
                   # --- CSV ---
                   csv_format='long'):
    """
    Pixel-wise H-score 分类 + 可视化 + 逐像素分类 CSV 导出。

    Parameters
    ----------
    marker_image : 2D ndarray 或 tiff 路径
        marker 强度图像 (建议先做背景扣除)。
    Hscore_threshold_annotation : dict or None
        参考 ROI 区域, 用于调取 ``Hscore_threshold`` 计算阈值, 形如
        ``{'Np': mask, 'Lp': mask, 'Mp': mask, 'Hp': mask}``;
        每个 value 可为单个 bool mask 或 mask 列表。
    tissue_annotation : dict or None
        待统计的区域标注 ``{name: mask 或 mask 列表}``; 非空时返回各区域
        pix H-score 柱状图 + 比例/面积表。
    tissue_mask : bool ndarray or None
        组织 mask (True=组织), 用于排除背景信号。
    thresholds : (T1, T2, T3) or None
        显式分层阈值, 顺序为 (Lp, Mp, Hp); None → 由
        ``Hscore_threshold_annotation`` 计算。
        分类规则: <T1→0, [T1,T2)→1, [T2,T3)→2, ≥T3→3。
    stat : str, {'mean','median','geomean','all'}
        选择用于阈值的统计量 (默认 'median')。
    delta : float or None
        分类着色图 (map) 显示前的高斯模糊 sigma (默认 None 不模糊)。
    kernel : int or tuple or None
        分类着色图 (map) 显示前的高斯模糊核大小 (默认 None)。
        统计柱状图/CSV 始终使用非高斯 (非模糊) 分类。
    save_dir : str, 输出目录。
    base_name : str, 输出文件前缀。
    map_* : 分类着色图相关样式 / 是否显示 / 保存。
    bar_* : 柱状图 + 表相关样式 / 是否显示 / 保存。
    pixel_area_um2 : float, 每像素面积 (µm²), 默认 25 (5µm × 5µm)。
    csv_format : str, 'long' (逐像素行 x,y,intensity,score,category)
                或 'matrix' (2D score 矩阵 CSV)。

    Returns
    -------
    dict
        {
          'thresholds'  : (T1, T2, T3),
          'stat'        : 所选统计量 ('mean'/'median'/'geomean'/'all'),
          'stats'       : Hscore_threshold 返回的完整统计 dict 或 None,
          'scores_2d'   : 非高斯分类 uint8 ndarray (H, W) (bargraph/CSV 用),
          'scores_2d_blur' : 高斯模糊分类 uint8 ndarray (H, W) (map 显示用),
          'csv_path'    : str,
          'score_tiff_path' : str,
          'fig_map'     : matplotlib Figure or None,
          'results'     : {region: {...}} or None (tissue_annotation 非空时),
          'fig_bar'     : matplotlib Figure or None,
          'categories'  : {0:'NP',1:'LP',2:'MP',3:'HP'},
        }
    """
    marker_image = _load(marker_image)
    os.makedirs(save_dir, exist_ok=True)

    # ---- 1) 阈值 ----
    stats = None
    if thresholds is None:
        if not Hscore_threshold_annotation:
            raise ValueError(
                "必须提供 thresholds 或 Hscore_threshold_annotation "
                "(包含 Np/Lp/Mp/Hp 参考 ROI)。")
        stats = Hscore_threshold(
            marker_image,
            Np=Hscore_threshold_annotation.get('Np'),
            Lp=Hscore_threshold_annotation.get('Lp'),
            Mp=Hscore_threshold_annotation.get('Mp'),
            Hp=Hscore_threshold_annotation.get('Hp'),
            tissue_mask=tissue_mask,
            stat=stat)
        thresholds = stats['thresholds']

    thresholds = tuple(thresholds)
    if len(thresholds) != 3 or any(t is None for t in thresholds):
        raise ValueError(f"thresholds 必须为 (T1, T2, T3) 三个非 None 值, 得到 {thresholds}")

    # ---- 2) 分类 ----
    # 非高斯分类: 用于 bargraph 统计与逐像素 CSV
    scores_2d = _classify(marker_image, thresholds,
                          blur_sigma=None, blur_kernel=None)
    # 高斯模糊分类: 仅用于 map 显示 (delta=sigma, kernel=核大小)
    if kernel is not None and delta not in (None, 0):
        scores_2d_blur = _classify(marker_image, thresholds,
                                   blur_sigma=delta, blur_kernel=kernel)
    else:
        scores_2d_blur = scores_2d

    # ---- 3) 逐像素分类 CSV + score tiff ----
    csv_path = os.path.join(save_dir, f"{base_name}_pixels.csv")
    score_tiff_path = os.path.join(save_dir, f"{base_name}_scores.tiff")

    if csv_format == 'matrix':
        np.savetxt(csv_path, scores_2d, delimiter=',', fmt='%d')
    else:  # 'long'
        if tissue_mask is not None:
            valid = np.asarray(tissue_mask, dtype=bool)
        else:
            valid = np.ones_like(scores_2d, dtype=bool)
        ys, xs = np.nonzero(valid)
        ints = marker_image[valid]
        scs = scores_2d[valid]
        cats = _CAT_ARR[scs]
        # 结构化数组一次性写出 (快速, 适合数百万像素)
        rec = np.empty(len(xs), dtype=[
            ('x', np.int64), ('y', np.int64),
            ('intensity', np.float64), ('score', np.int64),
            ('category', 'U2')])
        rec['x'] = xs
        rec['y'] = ys
        rec['intensity'] = ints
        rec['score'] = scs
        rec['category'] = cats
        np.savetxt(csv_path, rec, delimiter=',',
                   fmt=['%d', '%d', '%.4f', '%d', '%s'],
                   header='x,y,intensity,score,category', comments='')

    tifffile.imwrite(score_tiff_path, scores_2d.astype(np.uint8), compression='lzw')

    # ---- 4) 分类着色图 (colorbar 在图外右侧) ----
    fig_map = None
    if map_show or map_save:
        cmap = ListedColormap(map_cmap) if isinstance(map_cmap, (list, tuple)) else map_cmap
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], ncolors=5)
        disp = scores_2d_blur.astype(np.int8)
        if tissue_mask is not None:
            disp[~np.asarray(tissue_mask, dtype=bool)] = -1  # 背景 → 白

        fig_map, ax = plt.subplots(figsize=map_figsize)
        im = ax.imshow(disp, cmap=cmap, norm=norm, interpolation='nearest')

        ax.xaxis.set_major_locator(MultipleLocator(map_tick_step))
        ax.yaxis.set_major_locator(MultipleLocator(map_tick_step))
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.set_xlabel(xlabel, fontsize=axis_label_fontsize, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=axis_label_fontsize, fontweight='bold')
        ax.grid(True, alpha=0.2, linewidth=2.0)
        if map_title is not None:
            ax.set_title(map_title, fontsize=46, fontweight='bold')

        if map_colorbar:
            cbar = fig_map.colorbar(im, ax=ax, fraction=0.035, pad=0.02,
                                    ticks=[0, 1, 2, 3])
            cbar.ax.set_yticklabels(map_colorbar_labels)
            cbar.set_label(map_colorbar_label, fontsize=34, fontweight='bold',
                           labelpad=10)
            cbar.ax.tick_params(labelsize=24)

        fig_map.tight_layout()
        if map_show:
            plt.show()
        if map_save:
            map_path = os.path.join(save_dir, f"{base_name}_map.png")
            fig_map.savefig(map_path, dpi=map_dpi, bbox_inches='tight',
                            facecolor='white')
            print(f"分类着色图已保存到: {map_path}")

    # ---- 5) 各区域 pix H-score 柱状图 + 表 (tissue_annotation 非空时) ----
    fig_bar = None
    results = None
    if tissue_annotation:
        region_names = bar_order if bar_order is not None else list(tissue_annotation.keys())
        results = {}
        for name in region_names:
            roi = _as_mask(tissue_annotation[name], marker_image.shape)
            sel = roi & (np.asarray(tissue_mask, dtype=bool)
                         if tissue_mask is not None else np.ones_like(roi, bool))
            res = _hscore_from_scores(scores_2d, sel, thresholds)
            res['roi_px'] = int(roi.sum()) if roi is not None else 0
            res['valid_px'] = int(sel.sum())
            results[name] = res

        fig_bar = _bargraph_table(
            region_names, results, pixel_area_um2=pixel_area_um2,
            region_colors=region_colors,
            figsize=bar_figsize,
            overall_sel=(np.asarray(tissue_mask, dtype=bool)
                         if tissue_mask is not None else None),
            scores_2d=scores_2d, thresholds=thresholds)
        if bar_show:
            plt.show()
        if bar_save:
            bar_path = os.path.join(save_dir, f"{base_name}_bargraph.png")
            fig_bar.savefig(bar_path, dpi=bar_dpi, bbox_inches='tight',
                            facecolor='white')
            print(f"柱状图 + 表已保存到: {bar_path}")

    print(f"逐像素分类 CSV 已保存到: {csv_path}")
    print(f"score 图已保存到: {score_tiff_path}")

    return {
        'thresholds': thresholds,
        'stat': stat,
        'stats': stats,
        'scores_2d': scores_2d,
        'scores_2d_blur': scores_2d_blur,
        'csv_path': csv_path,
        'score_tiff_path': score_tiff_path,
        'fig_map': fig_map,
        'results': results,
        'fig_bar': fig_bar,
        'categories': dict(CATEGORY_NAMES),
    }


# ============================================================
# 柱状图 + 比例/面积表 (publication figure)
# ============================================================

def _bargraph_table(region_names, results, pixel_area_um2=25,
                    region_colors=None, figsize=(6, 6.5),
                    overall_sel=None, scores_2d=None, thresholds=(0, 0, 0)):
    """柱状图 (上) + 各区域 NP/LP/MP/HP 比例与面积表 (下) 的复合图。"""
    if region_colors is None:
        default_colors = ['#e8481e', '#808080', '#cc66e6', '#1f77b4',
                          '#2ca02c', '#ff7f0e', '#d62728', '#17becf']
        region_colors = {n: default_colors[i % len(default_colors)]
                         for i, n in enumerate(region_names)}

    bar_width = 0.62
    tick_fontsize = 15
    value_fontsize = 16
    table_fontsize = 14
    axis_label_fontsize = 18

    names = region_names
    scores = [results[n]['pix_h'] for n in names]
    percs = [[results[n]['percents'][k] for k in (0, 1, 2, 3)] for n in names]
    areas = [results[n]['valid_px'] * pixel_area_um2 / 1e6 for n in names]
    colors = [region_colors[n] for n in names]
    n = len(names)
    x = np.arange(n)

    overall_hscore = None
    if scores_2d is not None and overall_sel is not None:
        overall_hscore = _hscore_from_scores(scores_2d, overall_sel, thresholds)['pix_h']

    fig = plt.figure(figsize=figsize)
    ax_bar = fig.add_axes([0.13, 0.54, 0.84, 0.41])
    ax_tab = fig.add_axes([0.13, 0.06, 0.84, 0.40])

    # ---- 上: 柱状图 ----
    ax_bar.bar(x, scores, width=bar_width, color=colors,
               edgecolor='white', linewidth=0.8, zorder=3)
    for xi, s in zip(x, scores):
        ax_bar.text(xi, s + max(scores) * 0.02, f'{s:.1f}',
                    ha='center', va='bottom', fontsize=value_fontsize,
                    fontweight='bold')
    if overall_hscore is not None:
        ax_bar.axhline(overall_hscore, color='#444444', ls='--', lw=1.4, zorder=2)
        ax_bar.text(n - 0.5, overall_hscore + max(scores) * 0.015,
                    f'Overall {overall_hscore:.1f}', ha='right', va='bottom',
                    fontsize=11, color='#444444', style='italic')
    ax_bar.set_ylabel('pix H-Score', fontsize=axis_label_fontsize,
                      fontweight='bold')
    ax_bar.set_ylim(0, max(scores) * 1.28)
    ax_bar.set_xlim(-0.5, n - 0.5)
    ax_bar.set_xticks([])
    ax_bar.tick_params(axis='y', labelsize=tick_fontsize)
    for lb in ax_bar.get_yticklabels():
        lb.set_fontweight('bold')
    ax_bar.grid(True, axis='y', ls='--', alpha=0.45)
    ax_bar.set_axisbelow(True)
    for sp in ax_bar.spines.values():
        sp.set_linewidth(0.8)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    # ---- 下: 比例 / 面积表 ----
    ROW_LABELS = [r'$N_{P}$', r'$L_{P}$', r'$M_{P}$', r'$H_{P}$', r'Area']
    n_rows = 5
    ax_tab.set_xlim(-0.5, n - 0.5)
    ax_tab.set_ylim(-0.55, n_rows)
    ax_tab.axis('off')
    amax = max(areas)
    for j, name in enumerate(names):
        rgbc = to_rgba(region_colors[name])[:3]
        for i in range(n_rows):
            y = n_rows - 1 - i
            if i < 4:
                val = percs[j][i]
                alpha = 0.10 + 0.80 * (val / 100.0)
                fc = (*rgbc, alpha)
                txt = f'{val:.1f}'
                tcol = 'black' if alpha < 0.6 else 'white'
            else:
                alpha = 0.10 + 0.55 * (areas[j] / amax)
                fc = (0.4, 0.4, 0.4, alpha)
                txt = f'{areas[j]:.1f}'
                tcol = 'black' if alpha < 0.55 else 'white'
            ax_tab.add_patch(Rectangle((j - 0.5, y), 1.0, 1.0, facecolor=fc,
                                       edgecolor='white', linewidth=1.2))
            ax_tab.text(j, y + 0.5, txt, ha='center', va='center',
                        fontsize=table_fontsize, fontweight='bold', color=tcol)
    for i, lab in enumerate(ROW_LABELS):
        ax_tab.text(-0.62, n_rows - 1 - i + 0.5, lab, ha='right', va='center',
                    fontsize=table_fontsize + 2, fontweight='bold')
    for j, name in enumerate(names):
        ax_tab.text(j, -0.12, name.replace(' ', '\n'), ha='center', va='top',
                    fontsize=table_fontsize, fontweight='bold',
                    color=region_colors[name])

    return fig
