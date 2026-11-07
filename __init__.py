"""
IMC-HistoTools — digital pathology analysis toolkit for the Selenium (Se) IMC paper.

Provides:
  * virtual_IHC            : Beer-Lambert virtual IHC (brightfield-like) rendering
  * virtual_IHC_ROI        : local-region rendering that keeps original pixel coords
  * virtual_IHC_annotation : Qupath geojson -> ROI masks
  * Hscore_threshold       : median-intensity thresholds from Lp/Mp/Hp reference ROIs
  * pix_Hscore_plot_map    : classification color map (map)
  * pix_Hscore_plot_bar    : per-region pix H-score bar chart + proportion table
"""

from .virtual_IHC import virtual_IHC, virtual_IHC_ROI
from .annotation_region import virtual_IHC_annotation, Hscore_threshold
from .pix_Hscore import (pix_Hscore_plot_map, pix_Hscore_plot_bar,
                         SCORE_COLORS, BAR_ORDER)

__version__ = "0.2.0"

__all__ = [
    "virtual_IHC",
    "virtual_IHC_ROI",
    "virtual_IHC_annotation",
    "Hscore_threshold",
    "pix_Hscore_plot_map",
    "pix_Hscore_plot_bar",
    "SCORE_COLORS",
    "BAR_ORDER",
]
