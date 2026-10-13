# IMC-HistoTools

Digital pathology analysis tools for the **Selenium (Se) Imaging Mass Cytometry (IMC)**
paper. This package bundles the reusable functions behind:

1. **Virtual IHC rendering** — Beer–Lambert synthesis of IMC channels into a
   brightfield-like (IHC) RGB image.
2. **Qupath ROI annotation** — rasterization of `.geojson` annotations into ROI masks.
3. **Pixel-wise H-score** — threshold-based classification of every pixel into
   `NP / LP / MP / HP` (0 / 1 / 2 / 3), with per-pixel CSV export and publication
   figures (classification map, bar chart + table).

## Installation

```bash
pip install imc-histotools
```

Or install from source:

```bash
git clone https://github.com/gjhan100/IMC-HistoTools.git
cd IMC-HistoTools
pip install -e .
```

## Quick start

```python
from IMC_HistoTools import virtual_IHC, virtual_IHC_ROI
from IMC_HistoTools import virtual_IHC_annotation, Hscore_threshold
from IMC_HistoTools import pix_Hscore_map
```

### 1. Virtual IHC

```python
# Blue = DNA (nuclear), Brown = Se-tagged membrane marker
virtual_IHC(Blue_image=dna, Brown_image=se_cd45,
            blue_name="DNA", brown_name="CD45 (Se)",
            save_dir="output/")

# local region, keeping original pixel coordinates
virtual_IHC_ROI(Blue_image=dna, Brown_image=se_cd45,
                X0=2250, X1=3000, Y0=1250, Y1=2000, TICK=250,
                save_dir="output/")
```

### 2. Qupath geojson → ROI masks

```python
ann = virtual_IHC_annotation(marker_image, "Qpath_annotations.geojson",
                             region_order=["Normal mucosa", "Tumor center"])
region_masks = ann["masks"]   # {name: bool ndarray}
```

### 3. Pixel-wise H-score

```python
# reference ROIs for calibration: Np / Lp / Mp / Hp (mask or list of masks)
result = pix_Hscore_map(
    marker_image=se_ecadherin,
    Hscore_threshold_annotation={"Np": neg_roi, "Lp": low_roi,
                                 "Mp": mod_roi, "Hp": high_roi},
    tissue_annotation=region_masks,     # non-empty -> bar chart + table
    tissue_mask=tissue_mask,
    save_dir="output/",
)

result["csv_path"]      # per-pixel classification CSV (x, y, intensity, score, category)
result["scores_2d"]     # 0/1/2/3 score map
result["thresholds"]    # (T1, T2, T3)
```

## Modules

| Module              | Functions                                             |
| ------------------- | ----------------------------------------------------- |
| `virtual_IHC.py`    | `virtual_IHC`, `virtual_IHC_ROI`                      |
| `annotation_region.py` | `virtual_IHC_annotation`, `Hscore_threshold`        |
| `pix_Hscore.py`     | `pix_Hscore_map`                                      |

## License

MIT
