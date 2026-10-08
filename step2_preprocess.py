"""
الخطوة 2: المعالجة المسبقة.
-------------------------------------------------
ماذا نفعل؟
  - نملأ القيم المفقودة (مثل فجوات السحب) بأقرب قيمة صالحة.
  - نقصّ القيم الشاذة المتطرفة (percentile 0.5 - 99.5) للطبقات المستمرة.
  - نحفظ الطبقات النظيفة في data/processed/clean_layers.npz
ملاحظة: إعادة الإسقاط والقص وإعادة العينة تتم عادة قبل هذه الخطوة في GIS
        (QGIS / rasterio / GDAL)، ثم تُوضع النتائج في data/raw.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from config import LAYERS, OPTIONAL, PROC
from utils import load_layer

BINARY = {"wadi_mask", "urban", "flood_ref"}


def fill_nan_nearest(a):
    mask = np.isnan(a)
    if not mask.any():
        return a
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def main():
    clean = {}
    for name in LAYERS + OPTIONAL:
        arr = load_layer(name, required=name in LAYERS)
        if arr is None:
            continue
        if arr.ndim == 3:   # مكعب فائق الطيف: نعالج كل نطاق
            arr = np.stack([fill_nan_nearest(b) for b in arr])
        else:
            n_before = int(np.isnan(arr).sum())
            arr = fill_nan_nearest(arr)
            if name not in BINARY:
                lo, hi = np.percentile(arr, [0.5, 99.5])
                arr = np.clip(arr, lo, hi)
            if n_before:
                print(f"{name}: تم ملء {n_before} بكسل مفقود")
        clean[name] = arr.astype("float32")
    np.savez_compressed(PROC / "clean_layers.npz", **clean)
    print(f"تم حفظ {len(clean)} طبقة نظيفة -> {PROC / 'clean_layers.npz'}")


if __name__ == "__main__":
    main()
