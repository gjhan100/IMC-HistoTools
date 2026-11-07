"""
virtual_IHC.py — Virtual IHC rendering module (IMC-HistoTools)

Synthesizes a brightfield-like IHC RGB image from IMC multi-channel data using the
Beer-Lambert model. Supports per-channel gamma correction to match the color depth
and contrast of real IHC.

Derived from Se_IMC/Pseudo_brightfield.py with the following changes:
  * ``Virtual_IHC_show_save`` -> ``virtual_IHC`` (functionality unchanged)
  * colorbar label / fontsize / fontweight are exposed as function arguments
  * new ``virtual_IHC_ROI``: display a local region while keeping original pixel
    coordinates (not re-indexed from 0)

Usage:
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
# Color constants (Beer-Lambert absorption coefficients)
# ============================================================
BLK = [0.0, 0.0, 0.0]              # background (black)

BLUE = [9.147, 6.9215, 1.0]        # blue channel (nuclei / DNA)
RED  = [1.0,   15.8,   6.0]        # red channel
BROWN = [3.5,   5.3,   9.0]        # brown channel (membrane stains such as E-cadherin)


# ============================================================
# helper
# ============================================================

def _apply_gamma(img, gamma, ref_percentile=99.9):
    """Apply gamma correction while approximately preserving the intensity scale.

    Normalize to ``ref_percentile`` -> apply gamma -> rescale back to the
    original scale. ``gamma=1.0`` leaves the image unchanged.
    """
    if gamma == 1.0 or img is None or np.all(img == 0):
        return img
    ref = np.percentile(img[img > 0], ref_percentile) if np.any(img > 0) else 1.0
    if ref <= 0:
        return img
    img_norm = img / ref
    img_norm = np.clip(img_norm, 0, None)  # ensure non-negative
    return (img_norm ** gamma) * ref


# ============================================================
# main function
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
                bar_from_zero=True,
                blue_bar_label='Blue Intensity',
                red_bar_label='Red Intensity',
                brown_bar_label='Brown Intensity',
                bar_position='right',
                bar_pad=0.02,
                bar_fraction=0.02453,
                bar_shrink=0.85,
                bar_spacing=0.12,
                bar_tick_fontsize=12,
                bar_tick_fontweight='normal',
                bar_label=None,
                bar_label_fontsize=None,
                bar_label_fontweight=None):
    """
    Generate a virtual IHC image, display it, and save it.

    Beer-Lambert model:
      RGB[c] = BLK[c] + (1-BLK[c]) * exp(
          - k1*BLUE[c]*Blue_gamma  - k2*RED[c]*Red_gamma  - k3*BROWN[c]*Brown_gamma
      )

    Parameters
    ----------
    Blue_image  : 2D ndarray, blue channel (nuclei; must not be None)
    Red_image   : 2D ndarray or None, red channel
    Brown_image : 2D ndarray or None, brown channel (membrane stain)
    blue_name, red_name, brown_name : str, channel names
    title : str, figure title (default '' = hidden)
    k1, k2, k3 : float
        Global absorption coefficients controlling overall channel intensity.
    blue_gamma, red_gamma, brown_gamma : float
        Per-channel gamma correction (default 1.0).
    blue_threshold, red_threshold, brown_threshold : float
        Per-channel background-subtraction threshold: pixels below the threshold
        are set to 0 (default 0.0 = no subtraction). For example,
        Brown_threshold=0.1 is equivalent to np.where(Brown_image < 0.1, 0.0, Brown_image).
    sigma : float, Gaussian blur sigma
    kernel_size : tuple, Gaussian kernel size
    save_dir : str, output directory or full path (.png/.tiff/.pdf/.svg)
    figsize : tuple, figure size
    save_dpi : int, figure resolution (default 150)
    x_ticks, y_ticks : array-like or None
        Explicit axis ticks (takes precedence over tick_step), e.g. np.arange(0, 2001, 250).
    tick_step : int or None
        Axis tick interval (pixels). When x_ticks/y_ticks are None, ticks are
        generated with MultipleLocator(tick_step); default None = unset.
    xlim, ylim : tuple or None
        Axis display range (xmin, xmax) / (ymin, ymax). imshow defaults to
        -0.5 ~ W-0.5, which clips edge ticks (e.g. 2000); set (0, W) to show them.
    tick_fontsize : int, axis tick font size
    tick_fontweight : str, axis tick font weight
    xlabel : str, X-axis label (default '5μm / pixel')
    ylabel : str, Y-axis label (default '5μm / pixel')
    axis_label_fontsize : int, axis label font size
    axis_label_fontweight : str, axis label font weight
    title_fontsize : int, title font size
    title_fontweight : str, title font weight
    title_pad : float or None
        Spacing between title and image; default None uses the matplotlib default.
    grid : bool, whether to show the grid
    grid_linewidth : float, grid line width (default 0.5)
    spine_linewidth : float, outer spine line width (default 1.5)
    blue_bar, red_bar, brown_bar : bool
        Whether to show the per-channel Normalized Intensity [0,1] colorbar.
    bar_from_zero : bool
        Mapping of the colorbar 0 end (default True):
        - True  -> intensity = x*hi, 0 end = white (matches the image background);
        - False -> intensity = lo + x*(hi-lo), 0 end = minimum positive value (legacy).
    blue_bar_label, red_bar_label, brown_bar_label : str
        Per-channel colorbar labels (default 'Blue/Red/Brown Intensity').
    bar_label : str or None
        Overrides all channel colorbar labels when not None (default None = per-channel labels).
    bar_position : str, 'left' / 'right' / 'top' / 'bottom'
        Colorbar placement.
    bar_pad : float
        Spacing between the first colorbar and the image (default 0.02).
    bar_fraction : float
        Colorbar width (absolute figure fraction, default 0.02453). The three
        bars (blue/red/brown) always have this width, independent of figsize.
    bar_shrink : float
        Colorbar height (fraction of axes, default 0.85).
    bar_spacing : float
        Spacing increment between multiple colorbars (default 0.12).
    bar_tick_fontsize : int
        Colorbar tick font size.
    bar_tick_fontweight : str
        Colorbar tick font weight.
    bar_label_fontsize : int or None
        Colorbar label font size; None -> bar_tick_fontsize + 8.
    bar_label_fontweight : str or None
        Colorbar label font weight; None -> bar_tick_fontweight.
    """
    # ---- background subtraction (threshold): pixels below threshold set to 0 ----
    Blue_image  = np.where(Blue_image < blue_threshold, 0.0, Blue_image)    if Blue_image  is not None else None
    Red_image   = np.where(Red_image < red_threshold, 0.0, Red_image)      if Red_image   is not None else None
    Brown_image = np.where(Brown_image < brown_threshold, 0.0, Brown_image) if Brown_image is not None else None

    # ---- gamma correction ----
    Blue_image  = _apply_gamma(Blue_image,  blue_gamma)  if Blue_image  is not None else None
    Red_image   = _apply_gamma(Red_image,   red_gamma)   if Red_image   is not None else None
    Brown_image = _apply_gamma(Brown_image, brown_gamma) if Brown_image is not None else None

    # Colorbar mapping interval: lim percentiles from the (thresholded + gamma-corrected) image.
    #   lim=(0, 100) (default): lo = smallest positive pixel (after thresholding),
    #   hi = largest positive pixel; i.e. the 0 end of the colorbar = color of the
    #   smallest pixel after thresholding, and the 1 end = color of the largest pixel.
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

    # ---- input validation: determine height/width from available images (Blue > Red > Brown) ----
    if Blue_image is not None:
        height, width = Blue_image.shape
    elif Red_image is not None:
        height, width = Red_image.shape
    elif Brown_image is not None:
        height, width = Brown_image.shape
    else:
        raise ValueError("At least one non-None image (Blue / Red / Brown) is required.")

    if Blue_image is not None:
        print(f"Blue  image: {blue_name},  shape: {Blue_image.shape}, dtype: {Blue_image.dtype}")
    else:
        Blue_image = np.zeros((height, width))
        print(f"Blue  image: None -> replaced with an all-zero array")

    if Red_image is None:
        Red_image = np.zeros((height, width))
        print(f"Red   image: None -> replaced with an all-zero array")
    else:
        print(f"Red   image: {red_name},   shape: {Red_image.shape}, dtype: {Red_image.dtype}")

    if Brown_image is None:
        Brown_image = np.zeros((height, width))
        print(f"Brown image: None -> replaced with an all-zero array")
    else:
        print(f"Brown image: {brown_name}, shape: {Brown_image.shape}, dtype: {Brown_image.dtype}")

    # ---- Beer-Lambert color synthesis ----
    new_image = np.empty([height, width, 3], dtype=np.float64)

    for c in range(3):
        absorbance = (k1 * BLUE[c]  * Blue_image +
                      k2 * RED[c]   * Red_image +
                      k3 * BROWN[c] * Brown_image)
        new_image[:, :, c] = BLK[c] + (1.0 - BLK[c]) * np.exp(-absorbance)

    # clip to [0, 1]
    new_image = np.clip(new_image, 0.0, 1.0)

    # ---- Gaussian blur ----
    new_image_blur = cv2.GaussianBlur(
        new_image, ksize=kernel_size,
        sigmaX=sigma, sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT
    )

    # ---- display ----
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.imshow(new_image_blur)
    if title:
        # title_pad=None -> matplotlib default spacing; otherwise set explicitly
        if title_pad is None:
            ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
        else:
            ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight,
                         pad=title_pad)
    # Axis ticks: explicit x_ticks/y_ticks take precedence; otherwise use tick_step
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    elif tick_step is not None:
        ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    elif tick_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    # Axis range: make edge ticks (e.g. 2000) visible — imshow defaults to -0.5~W-0.5.
    # xlim=(xmin, xmax); ylim=(ymin, ymax) — the image y-axis runs top-to-bottom,
    # so ylim is passed reversed to set_ylim so that (0, 2000) puts 0 on top and 2000 at the bottom.
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
    # x -> x^gamma -> map to [lo, hi] -> Beer-Lambert
    def _make_stain_cmap(absorption, k_val, gamma_val, lo, hi,
                         bar_alpha=1.0, bar_depth=1.0, from_zero=False, n=256):
        """gamma + lim -> Beer-Lambert -> white-to-darkest stain colormap.

        from_zero=True -> intensity starts at 0 (0 end = white, matches image background);
        from_zero=False -> starts at lo (legacy; 0 end = minimum positive value).
        """
        x = np.linspace(0, 1, n)
        x_eff = x ** gamma_val
        intensity = x_eff * hi if from_zero else lo + x_eff * (hi - lo)
        absorbance = k_val * bar_depth * np.outer(intensity,
                                      np.array(absorption, dtype=np.float64))
        rgb = np.clip(np.exp(-absorbance), 0.0, 1.0)
        rgba = np.concatenate([rgb, np.full((n, 1), bar_alpha)], axis=1)
        return LinearSegmentedColormap.from_list('stain', rgba, N=n)

    channel_configs = [
        (blue_bar,  Blue_image,  _make_stain_cmap(BLUE,  k1, blue_gamma,  *_lo_hi['blue'],  blue_bar_alpha,  blue_bar_depth,  bar_from_zero),   blue_bar_label),
        (red_bar,   Red_image,   _make_stain_cmap(RED,   k2, red_gamma,   *_lo_hi['red'],   red_bar_alpha,   red_bar_depth,   bar_from_zero),   red_bar_label),
        (brown_bar, Brown_image, _make_stain_cmap(BROWN, k3, brown_gamma, *_lo_hi['brown'], brown_bar_alpha, brown_bar_depth, bar_from_zero),   brown_bar_label),
    ]

    # colorbar label style: None -> default (bar_tick_fontsize + 8 / bar_tick_fontweight)
    if bar_label_fontsize is None:
        bar_label_fontsize = bar_tick_fontsize + 8
    if bar_label_fontweight is None:
        bar_label_fontweight = bar_tick_fontweight

    # ---- explicit colorbar layout: uniform spacing between adjacent bars = bar_spacing ----
    # channels to display (channel_configs order: blue, red, brown)
    _visible = [(cmap, (bar_label if bar_label is not None else lbl))
                for (show, img, cmap, lbl) in channel_configs
                if show and img is not None and not np.all(img == 0)]

    # fix the main-axes layout first, then place colorbars manually relative to it
    plt.tight_layout()

    if _visible:
        def _draw_cbar(pos, cmap, lbl):
            cax = fig.add_axes(pos)
            sm = ScalarMappable(Normalize(vmin=0, vmax=1), cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax)
            # label: bar_label overrides all channels when not None; otherwise per-channel
            cbar.set_label(lbl, fontsize=bar_label_fontsize,
                           fontweight=bar_label_fontweight)
            cbar.ax.tick_params(labelsize=bar_tick_fontsize)
            for t in cbar.ax.get_yticklabels():
                t.set_fontweight(bar_tick_fontweight)

        ap = ax.get_position()               # main-axes position (figure fraction)
        n = len(_visible)
        # visual order: channels later in channel_configs are closer to the main axes
        order = list(reversed(_visible))

        if bar_position in ('right', 'left'):
            bar_w = bar_fraction             # colorbar width (absolute figure fraction)
            bar_h = bar_shrink * ap.height   # colorbar height
            pad = bar_pad * ap.width         # gap between main axes and first colorbar
            gap = bar_spacing * ap.width     # gap between adjacent colorbars (uniform)
            total = pad + n * bar_w + (n - 1) * gap
            new_w = max(ap.width - total, 0.05)
            y0 = ap.y0 + (ap.height - bar_h) / 2.0
            if bar_position == 'right':
                ax.set_position([ap.x0, ap.y0, new_w, ap.height])
                x, step = ap.x0 + new_w + pad, bar_w + gap
            else:  # left
                ax.set_position([ap.x0 + total, ap.y0, new_w, ap.height])
                x, step = ap.x0, -(bar_w + gap)
            for cmap, lbl in order:
                _draw_cbar([x, y0, bar_w, bar_h], cmap, lbl)
                x += step
        else:                                # top / bottom: vertical stacking
            bar_w = bar_shrink * ap.width
            bar_h = bar_fraction             # absolute figure fraction
            pad = bar_pad * ap.height
            gap = bar_spacing * ap.height
            total = pad + n * bar_h + (n - 1) * gap
            new_h = max(ap.height - total, 0.05)
            x0 = ap.x0 + (ap.width - bar_w) / 2.0
            if bar_position == 'bottom':
                ax.set_position([ap.x0, ap.y0, ap.width, new_h])
                y, step = ap.y0 - pad - bar_h, -(bar_h + gap)
            else:  # top
                ax.set_position([ap.x0, ap.y0 + total, ap.width, new_h])
                y, step = ap.y0 + ap.height + pad, (bar_h + gap)
            for cmap, lbl in order:
                _draw_cbar([x0, y, bar_w, bar_h], cmap, lbl)
                y += step

    plt.show()

    # ---- save the full figure (including title, colorbars, labels) ----
    if save_dir.lower().endswith(('.png', '.tiff', '.tif', '.jpg', '.jpeg', '.pdf', '.svg')):
        filepath = save_dir
    else:
        parts = [n for n in [blue_name, red_name, brown_name] if n]
        filename = '_'.join(parts) + '.png'
        filepath = os.path.join(save_dir, filename)

    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    fig.savefig(filepath, dpi=save_dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Figure saved to: {filepath}  (dpi={save_dpi})")

    # ---- optionally save the raw composite (no title/colorbar/axes/grid) ----
    if Save_raw:
        base = os.path.splitext(filepath)[0]
        raw_filepath = base + "_raw.tiff"
        image_to_save = (new_image_blur * 65535).astype(np.uint16)
        tifffile.imwrite(raw_filepath, image_to_save, compression='lzw')
        print(f"Raw image saved to: {raw_filepath}")

    return fig


# ============================================================
# ROI variant: display only a local region while keeping original pixel coordinates
# ============================================================

def virtual_IHC_ROI(Blue_image, Red_image, Brown_image,
                    X0=2250, X1=3000,
                    Y0=1250, Y1=2000,
                    TICK=250,
                    **kwargs):
    """
    Render a local ROI of the virtual IHC image, display it, and save it.

    Builds on ``virtual_IHC`` and uses ``xlim``/``ylim`` to show only the
    rectangular region [X0, X1) x [Y0, Y1), while the axis ticks keep the
    original pixel coordinates (not re-indexed from 0):
      X0, X1 : x pixel range [X0, X1)
      Y0, Y1 : y pixel range [Y0, Y1)
      TICK   : axis tick interval

    The Beer-Lambert model is identical to ``virtual_IHC``; all other parameters
    (gamma / colorbar / save options, etc.) are forwarded via **kwargs.

    Parameters
    ----------
    Blue_image  : 2D ndarray, blue channel (nuclei; must not be None)
    Red_image   : 2D ndarray or None, red channel
    Brown_image : 2D ndarray or None, brown channel (membrane stain)
    X0, X1 : int, x pixel range [X0, X1) (default 2250, 3000)
    Y0, Y1 : int, y pixel range [Y0, Y1) (default 1250, 2000)
    TICK   : int, axis tick interval (default 250)
    **kwargs : forwarded to ``virtual_IHC`` (title, k1/k2/k3, gamma, lim,
        sigma, save_dir, figsize, save_dpi, Save_raw, tick/axis styling,
        colorbar styling, etc.).
    """
    # explicit ticks preserve the original pixel coordinates (2250, 2500, 2750, 3000 ...)
    x_ticks = np.arange(X0, X1 + 1, TICK)
    y_ticks = np.arange(Y0, Y1 + 1, TICK)

    return virtual_IHC(Blue_image, Red_image, Brown_image,
                       xlim=(X0, X1),
                       ylim=(Y0, Y1),
                       x_ticks=x_ticks,
                       y_ticks=y_ticks,
                       **kwargs)
