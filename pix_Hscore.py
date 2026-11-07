"""
pix_Hscore.py — Pixel-wise H-score classification and visualization (IMC-HistoTools)

Plotting functions:
  - ``pix_Hscore_plot_map`` : classification color map (map)
  - ``pix_Hscore_plot_bar`` : bar chart + proportion table

Corresponds to the core logic of 2610/261001_CRC_Pix_Hscore.ipynb and
2610/261003_Pix_Hscore_histopathology.ipynb.
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
# constants
# ============================================================
SCORE_COLORS = ['#FFFFFF', '#2166AC', '#67AB49', '#FDBF2C', '#D73027']
SCORE_LABELS = ['0 (Neg)', '1 (Low)', '2 (Mod)', '3 (High)']
CATEGORY_NAMES = {0: 'NP', 1: 'LP', 2: 'MP', 3: 'HP'}
_CAT_ARR = np.array(['NP', 'LP', 'MP', 'HP'])

BAR_ORDER = ['Tumor center', 'Normal mucosa', 'Satellites', 'Invasive front']
BAR_COLORS = ["#385aee", "#ca81ec", "#858484", "#ec360d"]


# ============================================================
# helper
# ============================================================

def _load(img):
    """Return an ndarray unchanged; treat str/Path as a tiff path to read."""
    if isinstance(img, (str, Path)):
        return np.asarray(tifffile.imread(str(img))).astype(np.float32)
    return np.asarray(img)


def _classify(marker_image, thresholds, blur_sigma=None, blur_kernel=None):
    """Threshold into levels (0/1/2/3). Optional Gaussian blur.

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
    """Compute pix H-Score and per-level counts/proportions on the boolean subset sel."""
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
# plot 1: classification color map (map)
# ============================================================

def pix_Hscore_plot_map(marker_image, thresholds, tissue_mask=None,
                        blur_sigma=5, blur_kernel=(7, 7),
                        figsize=(12, 9), title=None,
                        cmap=SCORE_COLORS,
                        colorbar=True, colorbar_label='Score',
                        colorbar_labels=SCORE_LABELS,
                        tick_step=500,
                        xlabel='5μm / pixel', ylabel='5μm / pixel',
                        axis_label_fontsize=34, tick_fontsize=24,
                        grid=True, grid_linewidth=2.0, spine_linewidth=3.0,
                        save_path=None, dpi=150, show=True):
    """Draw a pixel-wise H-score classification color map.

    The displayed classification is obtained by Gaussian-blurring the marker image
    (``blur_sigma`` / ``blur_kernel``) before thresholding; statistics and CSV use
    the non-blurred classification.
    """
    marker_image = _load(marker_image)
    thresholds = tuple(thresholds)

    scores = _classify(marker_image, thresholds,
                       blur_sigma=blur_sigma, blur_kernel=blur_kernel)
    cmap_obj = ListedColormap(cmap) if isinstance(cmap, (list, tuple)) else cmap
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], ncolors=5)
    disp = scores.astype(np.int8)
    if tissue_mask is not None:
        disp[~np.asarray(tissue_mask, dtype=bool)] = -1

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(disp, cmap=cmap_obj, norm=norm, interpolation='nearest')

    ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize, fontweight='bold')
    if grid:
        ax.grid(True, alpha=0.2, linewidth=grid_linewidth)
    for sp in ax.spines.values():
        sp.set_linewidth(spine_linewidth)
    if title is not None:
        ax.set_title(title, fontsize=46, fontweight='bold')

    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(colorbar_labels)
        cbar.set_label(colorbar_label, fontsize=34, fontweight='bold', labelpad=10)
        cbar.ax.tick_params(labelsize=24)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    if show:
        plt.show()
    return fig


# ============================================================
# plot 2: bar chart + proportion table
# ============================================================

def pix_Hscore_plot_bar(results, bar_order=BAR_ORDER, region_colors=None,
                        figsize=(6, 6.5),
                        overall_hscore=None,
                        bar_title='pix H-scores', bar_title_pad=10,
                        title_fontsize=16, title_fontweight='bold',
                        tick_fontsize=16, tick_fontweight='bold',
                        value_fontsize=22, table_fontsize=20,
                        overall_fontsize=11,
                        save_path=None, dpi=150, show=True):
    """Draw per-region pix H-score bar chart (top) and NP/LP/MP/HP proportion table (bottom).

    ``results`` is ``{region: {'pix_h', 'percents', 'valid_px'}}``;
    bar order is controlled by ``bar_order``: missing regions are skipped and any
    extra regions are appended at the end.
    ``bar_title`` / ``bar_title_pad`` control the bar-chart title and its padding
    from the bars. All font sizes are adjustable: ``title_fontsize`` /
    ``title_fontweight`` (title), ``tick_fontsize`` / ``tick_fontweight`` (ticks),
    ``value_fontsize`` (bar values), ``table_fontsize`` (table),
    ``overall_fontsize`` (overall reference-line label).
    """
    names = [n for n in bar_order if n in results]
    names += [n for n in results if n not in names]
    if not names:
        raise ValueError("results is empty; cannot draw the bar chart")

    if region_colors is None:
        color_map = {n: BAR_COLORS[i % len(BAR_COLORS)]
                     for i, n in enumerate(names)}
    else:
        color_map = region_colors
    colors = [color_map[n] for n in names]

    scores = [results[n]['pix_h'] for n in names]
    percs = [[results[n]['percents'][k] for k in (0, 1, 2, 3)] for n in names]
    n = len(names)
    x = np.arange(n)

    bar_width = 0.55

    fig = plt.figure(figsize=figsize)
    ax_bar = fig.add_axes([0.13, 0.52, 0.84, 0.40])
    ax_tab = fig.add_axes([0.13, 0.06, 0.84, 0.40])

    # ---- top: bar chart ----
    bars = ax_bar.bar(x, scores, width=bar_width, color=colors,
                      edgecolor='white', linewidth=0.8, zorder=3)
    for bar, s in zip(bars, scores):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(scores) * 0.02, f'{s:.1f}',
                    ha='center', va='bottom', fontsize=value_fontsize,
                    fontweight=tick_fontweight)
    if overall_hscore is not None:
        ax_bar.axhline(overall_hscore, color='#444444', ls='--', lw=1.4, zorder=2)
        ax_bar.text(n - 0.5, overall_hscore + max(scores) * 0.015,
                    f'Overall {overall_hscore:.1f}', ha='right', va='bottom',
                    fontsize=overall_fontsize, color='#444444', style='italic')
    if bar_title:
        ax_bar.set_title(bar_title, fontsize=title_fontsize,
                         fontweight=title_fontweight, pad=bar_title_pad)
    ax_bar.set_ylim(0, max(scores) * 1.25)
    ax_bar.set_xlim(-0.5, n - 0.5)
    ax_bar.set_xticks([])
    ax_bar.tick_params(axis='y', labelsize=tick_fontsize)
    for lb in ax_bar.get_yticklabels():
        lb.set_fontweight(tick_fontweight)
    ax_bar.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax_bar.set_axisbelow(True)
    for sp in ax_bar.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)

    # ---- bottom: proportion table (top-to-bottom: Hp -> Mp -> Lp -> Np) ----
    ROW_ORDER = (3, 2, 1, 0)
    ROW_LABELS = [r'$H_{P}$', r'$M_{P}$', r'$L_{P}$', r'$N_{P}$']
    n_rows = 4
    ax_tab.set_xlim(-0.5, n - 0.5)
    ax_tab.set_ylim(-0.55, n_rows)
    ax_tab.axis('off')
    for j, name in enumerate(names):
        rgbc = to_rgba(color_map[name])[:3]
        for i in range(n_rows):
            y = n_rows - 1 - i
            val = percs[j][ROW_ORDER[i]]
            alpha = 0.10 + 0.80 * (val / 100.0)
            fc = (*rgbc, alpha)
            txt = f'{val:.1f}'
            tcol = 'black' if alpha < 0.6 else 'white'
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
                    color=color_map[name])

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    if show:
        plt.show()
    return fig
