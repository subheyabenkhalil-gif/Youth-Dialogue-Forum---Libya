"""
utils.py
دوال مساعدة: قراءة/حفظ الطبقات (npy أو GeoTIFF).
"""
import numpy as np
from config import RAW


def load_layer(name, required=True):
    """يقرأ طبقة من data/raw. يدعم .npy ويدعم .tif إن كانت rasterio مثبتة."""
    npy, tif = RAW / f"{name}.npy", RAW / f"{name}.tif"
    if npy.exists():
        return np.load(npy).astype("float32")
    if tif.exists():
        import rasterio  # pip install rasterio
        with rasterio.open(tif) as src:
            arr = src.read().astype("float32")
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
        return arr[0] if arr.shape[0] == 1 else arr
    if required:
        raise FileNotFoundError(f"الطبقة غير موجودة: {name} (.npy / .tif) في {RAW}")
    return None


def spatial_split(df, block, test_fraction, seed):
    """
    فصل مكاني بالكتل (spatial block split): نقسم الشبكة إلى كتل مربعة،
    ونحجز كتلًا كاملة للاختبار. هذا يقلل التسرب المكاني (spatial leakage)
    الذي ينتج عن توزيع بكسلات متجاورة بين التدريب والاختبار عشوائيًا.
    يرجع: (train_mask, test_mask, block_id)
    """
    block_id = (df["row"] // block) * 1000 + (df["col"] // block)
    ids = np.sort(block_id.unique())
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(len(ids) * test_fraction)))
    test_ids = rng.choice(ids, size=n_test, replace=False)
    test_mask = block_id.isin(test_ids).to_numpy()
    return ~test_mask, test_mask, block_id.to_numpy()
