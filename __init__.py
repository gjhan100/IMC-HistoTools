"""
IMC-HistoTools — digital pathology analysis toolkit for the Selenium (Se) IMC paper.

Provides:
  * virtual_IHC            : Beer-Lambert virtual IHC (brightfield-like) rendering
  * virtual_IHC_ROI        : local-region rendering that keeps original pixel coords
  * virtual_IHC_annotation : Qupath geojson -> ROI masks
  * Hscore_threshold       : median-intensity thresholds from Np/Lp/Mp/Hp reference ROIs
  * pix_Hscore_map         : pixel-wise H-score classification + CSV + figures
"""

from .virtual_IHC import virtual_IHC, virtual_IHC_ROI
from .annotation_region import virtual_IHC_annotation, Hscore_threshold
from .pix_Hscore import pix_Hscore_map

__version__ = "0.1.0"

__all__ = [
    "virtual_IHC",
    "virtual_IHC_ROI",
    "virtual_IHC_annotation",
    "Hscore_threshold",
    "pix_Hscore_map",
]
