"""
الخطوة 3: استخراج الخصائص (Feature Engineering).
-------------------------------------------------
الخصائص المشتقة:
  NDVI  = (NIR - Red) / (NIR + Red)          مؤشر الغطاء النباتي
  NDWI  = (Green - NIR) / (Green + NIR)      مؤشر المياه
  slope = الانحدار بالدرجات من DEM
  elevation, rain, wadi_dist (المسافة لأقرب مجرى), urban, s1_vv_pre
  (اختياري) sar_change = s1_post - s1_pre   فقط إذا USE_EVENT_SAR_AS_FEATURE=True
المتغير الهدف: flood_ref (مرجع للتحقق، لا يدخل كخاصية أبدًا).
المخرجات: data/processed/features.csv  (صف لكل بكسل)
"""
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from config import PROC, PIXEL_SIZE_M, USE_EVENT_SAR_AS_FEATURE


def main():
    L = np.load(PROC / "clean_layers.npz")
    eps = 1e-6
    ndvi = (L["s2_nir"] - L["s2_red"]) / (L["s2_nir"] + L["s2_red"] + eps)
    ndwi = (L["s2_green"] - L["s2_nir"]) / (L["s2_green"] + L["s2_nir"] + eps)
    gy, gx = np.gradient(L["dem"], PIXEL_SIZE_M)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    wadi_dist = distance_transform_edt(L["wadi_mask"] == 0) * PIXEL_SIZE_M

    feats = {
        "ndvi": ndvi, "ndwi": ndwi, "elevation": L["dem"], "slope": slope,
        "rain": L["rain"], "wadi_dist_m": wadi_dist, "urban": L["urban"],
        "s1_vv_pre": L["s1_vv_pre"],
    }
    if USE_EVENT_SAR_AS_FEATURE and "s1_vv_post" in L.files:
        feats["sar_change"] = L["s1_vv_post"] - L["s1_vv_pre"]
        print("تحذير: sar_change مفعّلة؛ تأكد أن مرجع الغمر مستقل عن SAR الحدث.")

    H, W = L["dem"].shape
    rr, cc = np.mgrid[0:H, 0:W]
    df = pd.DataFrame({k: v.ravel() for k, v in feats.items()})
    df["row"], df["col"] = rr.ravel(), cc.ravel()
    df["flood"] = L["flood_ref"].ravel().astype(int)
    df.to_csv(PROC / "features.csv", index=False)
    print(f"عدد البكسلات: {len(df)} | عدد الخصائص: {len(feats)}")
    print("الخصائص:", list(feats))
    print(f"نسبة الغمر في المرجع: {100 * df['flood'].mean():.1f}%")


if __name__ == "__main__":
    main()
