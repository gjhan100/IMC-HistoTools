"""
virtual_IHC.py — 虚拟 IHC 图像生成模块 (IMC-HistoTools)

用 Beer-Lambert 模型将 IMC 多通道数据合成为模拟 IHC 染色的 RGB 图像。
支持 per-channel gamma 校正以匹配真实 IHC 的颜色深浅对比。

对应 Se_IMC/Pseudo_brightfield.py，调整如下:
  * ``Virtual_IHC_show_save`` → ``virtual_IHC`` (功能不变)
  * colorbar ``cbar.set_label(...)`` 的 label / fontsize / fontweight
    提升为函数输入参数, 便于作图时直接修改
  * 新增 ``virtual_IHC_ROI``: 显示局部区域但保留原始像素坐标 (不从 0 重排)

用法:
  from IMC_HistoTools.virtual_IHC import virtual_IHC, virtual_IHC_ROI
  virtual_IHC(Blue_image=DNA, Brown_image=Se, ...)
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MultipleLocator
from matplotlib.cm import ScalarMappable
import tifffile


# ============================================================
# 颜色常量 (Beer-Lambert 吸收系数)
# ============================================================
BLK = [0.0, 0.0, 0.0]              # 背景 (黑色)

BLUE = [9.147, 6.9215, 1.0]        # 蓝色通道 (细胞核 / DNA)
RED  = [1.0,   15.8,   6.0]        # 红色通道
BROWN = [3.5,   5.3,   9.0]        # 棕色通道 (Ecadherin 等膜染色)


# ============================================================
# helper
# ============================================================

def _apply_gamma(img, gamma, ref_percentile=99.9):
    """对图像做 gamma 校正, 保持强度尺度大致不变.

    归一化到 ref_percentile → 应用 gamma → 缩放回原尺度.
    gamma=1.0 不做任何改变.
    """
    if gamma == 1.0 or img is None or np.all(img == 0):
        return img
    ref = np.percentile(img[img > 0], ref_percentile) if np.any(img > 0) else 1.0
    if ref <= 0:
        return img
    img_norm = img / ref
    img_norm = np.clip(img_norm, 0, None)  # 确保非负
    return (img_norm ** gamma) * ref


# ============================================================
# 主函数
# ============================================================

def virtual_IHC(Blue_image, Red_image, Brown_image,
                blue_name="Blue", red_name="Red", brown_name="Brown",
                title='',
                k1=0.001, k2=0.5, k3=0.0075,
                blue_gamma=1.0, red_gamma=1.0, brown_gamma=1.0,
                blue_threshold=0.0, red_threshold=0.0, brown_threshold=0.0,
                blue_lim=(0, 100),
                red_lim=(0, 100),
                brown_lim=(0, 100),
                sigma=1, kernel_size=(3, 3),
                save_dir='.', figsize=(24, 18),
                save_dpi=150,
                Save_raw=False,
                x_ticks=None, y_ticks=None,
                tick_step=None,
                xlim=None, ylim=None,
                tick_fontsize=10,
                tick_fontweight='normal',
                xlabel='5μm / pixel',
                ylabel='5μm / pixel',
                axis_label_fontsize=12,
                axis_label_fontweight='normal',
                title_fontsize=18,
                title_fontweight='bold',
                title_pad=None,
                grid=False,
                grid_linewidth=0.5,
                spine_linewidth=1.5,
                # --- colorbar ---
                blue_bar=False,
                red_bar=False,
                brown_bar=False,
                blue_bar_alpha=1.0,
                red_bar_alpha=1.0,
                brown_bar_alpha=1.0,
                blue_bar_depth=1.0,
                red_bar_depth=1.0,
                brown_bar_depth=1.0,
                bar_position='right',
                bar_pad=0.02,
                bar_tick_fontsize=9,
                bar_tick_fontweight='normal',
                bar_label='Normalized Intensity',
                bar_label_fontsize=None,
                bar_label_fontweight=None):
    """
    生成虚拟 IHC 图像，显示并保存。

    Beer-Lambert 模型:
      RGB[c] = BLK[c] + (1-BLK[c]) * exp(
          - k1*BLUE[c]*Blue_gamma  - k2*RED[c]*Red_gamma  - k3*BROWN[c]*Brown_gamma
      )

    Parameters
    ----------
    Blue_image  : 2D ndarray, 蓝色通道 (细胞核, 不可为 None)
    Red_image   : 2D ndarray or None, 红色通道
    Brown_image : 2D ndarray or None, 棕色通道 (膜染色)
    blue_name, red_name, brown_name : str, 各通道名称
    title : str, 图像标题 (默认 '' 不显示)
    k1, k2, k3 : float
        全局吸收系数, 控制各通道整体强度.
    blue_gamma, red_gamma, brown_gamma : float
        各通道 Gamma 校正 (默认 1.0).
    blue_threshold, red_threshold, brown_threshold : float
        各通道背景扣除阈值: 强度 < threshold 的像素置 0 (默认 0.0 不扣除).
        例如 Brown_threshold=0.1 等价于 np.where(Brown_image < 0.1, 0.0, Brown_image).
    sigma : float, 高斯模糊 sigma
    kernel_size : tuple, 高斯核大小
    save_dir : str, 输出目录或完整路径 (.png/.tiff/.pdf/.svg)
    figsize : tuple, 显示尺寸
    save_dpi : int, 保存 figure 的分辨率 (默认 150)
    x_ticks, y_ticks : array-like or None
        显式坐标轴刻度 (优先于 tick_step). 例如 np.arange(0, 2001, 250).
    tick_step : int or None
        坐标轴刻度间隔 (像素). 当 x_ticks/y_ticks 为 None 时, 用
        MultipleLocator(tick_step) 自动生成刻度; 默认 None 不设置.
    xlim, ylim : tuple or None
        坐标轴显示范围 (xmin, xmax) / (ymin, ymax). imshow 默认范围是
        -0.5 ~ W-0.5, 导致边界刻度(如 2000)被裁掉; 设为 (0, W) 即可显示.
    tick_fontsize : int, 坐标轴刻度字号
    tick_fontweight : str, 坐标轴刻度粗细
    xlabel : str, X 轴标签 (默认 '5μm / pixel')
    ylabel : str, Y 轴标签 (默认 '5μm / pixel')
    axis_label_fontsize : int, 坐标轴标签字号
    axis_label_fontweight : str, 坐标轴标签粗细
    title_fontsize : int, 标题字号
    title_fontweight : str, 标题粗细
    title_pad : float or None
        标题与图像之间的间距; 默认 None 使用 matplotlib 默认值.
    grid : bool, 是否显示网格
    grid_linewidth : float, 网格线宽度 (默认 0.5)
    spine_linewidth : float, 外边框线宽度 (默认 1.5)
    blue_bar, red_bar, brown_bar : bool
        是否显示各通道的 Normalized Intensity [0,1] colorbar.
    bar_position : str, 'left' / 'right' / 'top' / 'bottom'
        colorbar 放置位置.
    bar_pad : float
        colorbar 与图像的间距 (默认 0.02).
    bar_tick_fontsize : int
        colorbar 刻度字号.
    bar_tick_fontweight : str
        colorbar 刻度粗细.
    bar_label : str
        colorbar 标签文字 (默认 'Normalized Intensity').
    bar_label_fontsize : int or None
        colorbar 标签字号; None → bar_tick_fontsize + 8.
    bar_label_fontweight : str or None
        colorbar 标签粗细; None → bar_tick_fontweight.
    """
    # ---- 背景扣除 (threshold): 低于阈值的像素置 0 ----
    Blue_image  = np.where(Blue_image < blue_threshold, 0.0, Blue_image)    if Blue_image  is not None else None
    Red_image   = np.where(Red_image < red_threshold, 0.0, Red_image)      if Red_image   is not None else None
    Brown_image = np.where(Brown_image < brown_threshold, 0.0, Brown_image) if Brown_image is not None else None

    # ---- gamma 校正 ----
    Blue_image  = _apply_gamma(Blue_image,  blue_gamma)  if Blue_image  is not None else None
    Red_image   = _apply_gamma(Red_image,   red_gamma)   if Red_image   is not None else None
    Brown_image = _apply_gamma(Brown_image, brown_gamma) if Brown_image is not None else None

    # colorbar 用: 从 (阈值扣除 + gamma 校正后) 图像取 lim 百分位作为 [0,1] 映射区间.
    #   lim=(0, 100) (默认) 时 lo = 最小正像素 (阈值扣除后最小值), hi = 最大正像素,
    #   即 colorbar 的 0 端 = 阈值扣除后最小像素的颜色, 1 端 = 最大像素的颜色.
    def _get_lo_hi(img, lim):
        if img is None or np.all(img == 0):
            return 0.0, 1.0
        lo = np.percentile(img[img > 0], lim[0]) if np.any(img > 0) else 0.0
        hi = np.percentile(img[img > 0], lim[1]) if np.any(img > 0) else 1.0
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    _lo_hi = {}
    _lo_hi['blue']  = _get_lo_hi(Blue_image,  blue_lim)
    _lo_hi['red']   = _get_lo_hi(Red_image,   red_lim)
    _lo_hi['brown'] = _get_lo_hi(Brown_image, brown_lim)

    # ---- 输入校验: 从可用图像确定 height/width (优先级 Blue > Red > Brown) ----
    if Blue_image is not None:
        height, width = Blue_image.shape
    elif Red_image is not None:
        height, width = Red_image.shape
    elif Brown_image is not None:
        height, width = Brown_image.shape
    else:
        raise ValueError("至少需要提供一个非 None 的图像 (Blue / Red / Brown).")

    if Blue_image is not None:
        print(f"Blue  image: {blue_name},  shape: {Blue_image.shape}, dtype: {Blue_image.dtype}")
    else:
        Blue_image = np.zeros((height, width))
        print(f"Blue  image: None → 已用全零数组替代")

    if Red_image is None:
        Red_image = np.zeros((height, width))
        print(f"Red   image: None → 已用全零数组替代")
    else:
        print(f"Red   image: {red_name},   shape: {Red_image.shape}, dtype: {Red_image.dtype}")

    if Brown_image is None:
        Brown_image = np.zeros((height, width))
        print(f"Brown image: None → 已用全零数组替代")
    else:
        print(f"Brown image: {brown_name}, shape: {Brown_image.shape}, dtype: {Brown_image.dtype}")

    # ---- Beer-Lambert 颜色合成 ----
    new_image = np.empty([height, width, 3], dtype=np.float64)

    for c in range(3):
        absorbance = (k1 * BLUE[c]  * Blue_image +
                      k2 * RED[c]   * Red_image +
                      k3 * BROWN[c] * Brown_image)
        new_image[:, :, c] = BLK[c] + (1.0 - BLK[c]) * np.exp(-absorbance)

    # 裁剪到 [0, 1]
    new_image = np.clip(new_image, 0.0, 1.0)

    # ---- 高斯模糊 ----
    new_image_blur = cv2.GaussianBlur(
        new_image, ksize=kernel_size,
        sigmaX=sigma, sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT
    )

    # ---- 显示 ----
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.imshow(new_image_blur)
    if title:
        # title_pad=None → 用 matplotlib 默认间距; 否则显式设定
        if title_pad is None:
            ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
        else:
            ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight,
                         pad=title_pad)
    # 坐标轴刻度: 显式 x_ticks/y_ticks 优先; 否则用 tick_step 自动生成
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    elif tick_step is not None:
        ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    elif tick_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    # 坐标轴范围: 让边界刻度(如 2000)可见 —— imshow 默认范围是 -0.5~W-0.5
    # xlim=(xmin, xmax); ylim=(ymin, ymax) —— 图像 y 轴自上而下,
    # 故 ylim 反转传入 set_ylim, 保证 (0, 2000) 时 0 在上、2000 在下。
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim[1], ylim[0])
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight(tick_fontweight)
    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize,
                  fontweight=axis_label_fontweight)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize,
                  fontweight=axis_label_fontweight)
    if grid:
        ax.grid(True, alpha=0.3, linewidth=grid_linewidth)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)

    # ---- colorbar (Normalized Intensity [0,1] per channel) ----
    # x → x^gamma → 映射到 [lo, hi] → Beer-Lambert
    def _make_stain_cmap(absorption, k_val, gamma_val, lo, hi,
                         bar_alpha=1.0, bar_depth=1.0, n=256):
        """gamma + lim → Beer-Lambert → 白→最深染色 colormap."""
        x = np.linspace(0, 1, n)
        x_eff = x ** gamma_val
        intensity = lo + x_eff * (hi - lo)
        absorbance = k_val * bar_depth * np.outer(intensity,
                                      np.array(absorption, dtype=np.float64))
        rgb = np.clip(np.exp(-absorbance), 0.0, 1.0)
        rgba = np.concatenate([rgb, np.full((n, 1), bar_alpha)], axis=1)
        return LinearSegmentedColormap.from_list('stain', rgba, N=n)

    channel_configs = [
        (blue_bar,  Blue_image,  _make_stain_cmap(BLUE,  k1, blue_gamma,  *_lo_hi['blue'],  blue_bar_alpha,  blue_bar_depth),   blue_name),
        (red_bar,   Red_image,   _make_stain_cmap(RED,   k2, red_gamma,   *_lo_hi['red'],   red_bar_alpha,   red_bar_depth),    red_name),
        (brown_bar, Brown_image, _make_stain_cmap(BROWN, k3, brown_gamma, *_lo_hi['brown'], brown_bar_alpha, brown_bar_depth),  brown_name),
    ]

    # colorbar label 样式: None → 使用默认 (bar_tick_fontsize + 8 / bar_tick_fontweight)
    if bar_label_fontsize is None:
        bar_label_fontsize = bar_tick_fontsize + 8
    if bar_label_fontweight is None:
        bar_label_fontweight = bar_tick_fontweight

    # 累加 pad 偏移, 使多个 bar 不重叠
    pad_offset = 0.0
    for show, img, cmap, label in channel_configs:
        if not show or img is None or np.all(img == 0):
            continue

        sm = ScalarMappable(Normalize(vmin=0, vmax=1), cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax, location=bar_position,
                           pad=bar_pad + pad_offset,
                           fraction=0.035, shrink=0.85)
        cbar.set_label(bar_label,
                       fontsize=bar_label_fontsize,
                       fontweight=bar_label_fontweight)
        cbar.ax.tick_params(labelsize=bar_tick_fontsize)
        for t in cbar.ax.get_yticklabels():
            t.set_fontweight(bar_tick_fontweight)
        pad_offset += 0.12  # 下一个 bar 向外偏移

    plt.tight_layout()
    plt.show()

    # ---- 保存完整 figure (含 title, colorbar, 标注) ----
    if save_dir.lower().endswith(('.png', '.tiff', '.tif', '.jpg', '.jpeg', '.pdf', '.svg')):
        filepath = save_dir
    else:
        parts = [n for n in [blue_name, red_name, brown_name] if n]
        filename = '_'.join(parts) + '.png'
        filepath = os.path.join(save_dir, filename)

    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    fig.savefig(filepath, dpi=save_dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Figure 已保存到: {filepath}  (dpi={save_dpi})")

    # ---- 额外保存 raw 合成图 (无任何标注: 无 title/colorbar/坐标轴/网格) ----
    if Save_raw:
        base = os.path.splitext(filepath)[0]
        raw_filepath = base + "_raw.tiff"
        image_to_save = (new_image_blur * 65535).astype(np.uint16)
        tifffile.imwrite(raw_filepath, image_to_save, compression='lzw')
        print(f"Raw 图像已保存到: {raw_filepath}")

    return fig


# ============================================================
# ROI 版本: 仅显示局部区域, 但保留原始像素坐标 (不从 0 重新开始)
# ============================================================

def virtual_IHC_ROI(Blue_image, Red_image, Brown_image,
                    X0=2250, X1=3000,
                    Y0=1250, Y1=2000,
                    TICK=250,
                    **kwargs):
    """
    生成虚拟 IHC 图像的局部 ROI 区域，显示并保存。

    在 ``virtual_IHC`` 基础上, 通过 ``xlim``/``ylim`` 仅显示矩形区域
    [X0, X1) x [Y0, Y1), 但坐标轴刻度保留原始像素坐标 (不从 0 重新开始):
      X0, X1 : x 像素范围 [X0, X1)
      Y0, Y1 : y 像素范围 [Y0, Y1)
      TICK   : 坐标刻度间隔

    Beer-Lambert 模型与 ``virtual_IHC`` 完全一致; 其余参数 (gamma / colorbar /
    保存选项等) 经 **kwargs 透传给 ``virtual_IHC``。

    Parameters
    ----------
    Blue_image  : 2D ndarray, 蓝色通道 (细胞核, 不可为 None)
    Red_image   : 2D ndarray or None, 红色通道
    Brown_image : 2D ndarray or None, 棕色通道 (膜染色)
    X0, X1 : int, x 像素范围 [X0, X1) (默认 2250, 3000)
    Y0, Y1 : int, y 像素范围 [Y0, Y1) (默认 1250, 2000)
    TICK   : int, 坐标刻度间隔 (默认 250)
    **kwargs : 其余参数透传给 ``virtual_IHC`` (title, k1/k2/k3, gamma, lim,
        sigma, save_dir, figsize, save_dpi, Save_raw, tick/axis 样式,
        colorbar 样式等)。
    """
    # 显式坐标刻度: 保留原始像素坐标 (2250, 2500, 2750, 3000 ...)
    x_ticks = np.arange(X0, X1 + 1, TICK)
    y_ticks = np.arange(Y0, Y1 + 1, TICK)

    return virtual_IHC(Blue_image, Red_image, Brown_image,
                       xlim=(X0, X1),
                       ylim=(Y0, Y1),
                       x_ticks=x_ticks,
                       y_ticks=y_ticks,
                       **kwargs)
