"""
الخطوة 0: إنشاء بيانات تجريبية (Synthetic) لاختبار أن الكود يعمل.
=====================================================================
تنبيه: هذه ليست بيانات درنة الحقيقية، وأي أرقام تخرج منها ليست نتائج علمية.
للعمل على بيانات حقيقية: ضع الطبقات في data/raw بنفس الأسماء الموجودة في config.LAYERS
وتجاوز هذه الخطوة.

تشغيل:  python step0_make_demo_data.py            (بدون 813)
        python step0_make_demo_data.py --with-hyper   (يولّد مكعب فائق الطيف تجريبي لاختبار الخطوة 8)
"""
import sys
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from config import RAW, GRID, SEED


def smooth(rng, sigma, scale=1.0):
    return gaussian_filter(rng.normal(size=GRID), sigma) * scale


def z(a):
    return (a - a.mean()) / (a.std() + 1e-9)


def main(with_hyper=False):
    rng = np.random.default_rng(SEED)
    H, W = GRID
    rows, cols = np.mgrid[0:H, 0:W]

    # --- شبكة الأودية: مجرى متعرج ---
    center = 100 + 30 * np.sin(cols / 25.0)
    wadi_mask = (np.abs(rows - center) < 1.0).astype("float32")
    dist_wadi = distance_transform_edt(wadi_mask == 0)

    # --- DEM: يرتفع بعيدًا عن الوادي وينخفض نحو الشرق ---
    dem = 20 + 0.8 * dist_wadi + 0.15 * (W - cols) + smooth(rng, 8, 60)
    # --- الأمطار: أعلى في الغرب ---
    rain = 90 + 0.25 * (W - cols) + smooth(rng, 15, 120)
    # --- العمران: قرب الوادي في المناطق المنخفضة ---
    urban = ((dist_wadi < 25) & (z(dem) < 0.2) & (smooth(rng, 6, 10) > 0)).astype("float32")

    # --- الانحدار (لتوليد الغمر فقط) ---
    gy, gx = np.gradient(dem)
    slope = np.degrees(np.arctan(np.hypot(gx, gy) / 30.0))

    # --- مرجع الغمر (مولّد من عوامل فيزيائية + ضجيج) ---
    score = (-1.3 * z(dist_wadi) - 1.0 * z(dem) + 0.8 * z(rain)
             - 0.4 * z(slope) + rng.normal(scale=0.6, size=GRID))
    flood = (score > np.quantile(score, 0.85)).astype("float32")

    # --- Sentinel-2 (قبل الحدث): النبات والرطوبة يتبعان الوادي والعمران،
    #     ولا يعتمدان على مرجع الغمر مباشرة (لتجنب التسرب في البيانات التجريبية) ---
    veg = np.clip(0.3 + 0.004 * dist_wadi + smooth(rng, 5, 0.15), 0, 0.9) * (1 - urban * 0.8)
    moist = smooth(rng, 6, 0.04)
    red = np.clip(0.25 - 0.15 * veg + smooth(rng, 2, 0.02), 0.01, 1)
    nir = np.clip(0.20 + 0.45 * veg + smooth(rng, 2, 0.03), 0.01, 1)
    green = np.clip(0.15 + 0.05 * veg + moist + smooth(rng, 2, 0.01), 0.01, 1)
    swir = np.clip(0.30 - 0.12 * veg - moist + smooth(rng, 2, 0.02), 0.01, 1)
    # فجوات سحب في Sentinel-2 لاختبار معالجة القيم المفقودة
    for arr in (red, nir, green, swir):
        arr[20:35, 150:170] = np.nan

    # --- Sentinel-1 VV (dB): قبل الحدث ثم أثناءه (ينخفض فوق الماء) ---
    s1_pre = -11 + smooth(rng, 3, 2.0) - 1.5 * urban
    s1_post = s1_pre - 6.0 * flood + rng.normal(scale=0.7, size=GRID)

    layers = dict(dem=dem, rain=rain, wadi_mask=wadi_mask, urban=urban,
                  s1_vv_pre=s1_pre, s1_vv_post=s1_post,
                  s2_green=green, s2_red=red, s2_nir=nir, s2_swir=swir,
                  flood_ref=flood)

    if with_hyper:
        n_bands = 40
        base = np.stack([smooth(rng, 4, 1.0) for _ in range(n_bands)])
        layers["hyper813"] = (base + 0.6 * veg + 4.0 * moist).astype("float32")  # لا يعتمد على الغمر

    for name, arr in layers.items():
        np.save(RAW / f"{name}.npy", arr.astype("float32"))
    print(f"تم حفظ {len(layers)} طبقة تجريبية في {RAW}")
    print("تذكير: هذه بيانات Synthetic للاختبار فقط.")


if __name__ == "__main__":
    main(with_hyper="--with-hyper" in sys.argv)
