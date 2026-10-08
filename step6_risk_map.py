"""
الخطوة 6: إنتاج خريطة الاحتمالية وفئات الخطر.
-------------------------------------------------
- نتنبأ باحتمالية الغمر لكل بكسل.
- نحوّل الاحتمال إلى 4 فئات: Low / Medium / High / Severe
  (العتبات في config.RISK_BINS وقابلة للمعايرة).
المخرجات: outputs/risk_prob.npy, outputs/risk_class.npy, outputs/step6_risk_map.png,
          outputs/step6_class_area.csv
تنبيه: خريطة تجريبية/بحثية وليست إنذار طوارئ تشغيليًا.
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from config import PROC, OUT, RISK_BINS, RISK_LABELS, PIXEL_SIZE_M

META = ["row", "col", "flood"]


def main():
    df = pd.read_csv(PROC / "features.csv")
    bundle = joblib.load(OUT / "baseline_model.joblib")
    model, feats = bundle["model"], bundle["features"]

    proba = model.predict_proba(df[feats])[:, 1]
    H, W = df["row"].max() + 1, df["col"].max() + 1
    prob_map = proba.reshape(H, W)
    class_map = np.digitize(prob_map, RISK_BINS)     # 0..3
    np.save(OUT / "risk_prob.npy", prob_map)
    np.save(OUT / "risk_class.npy", class_map)

    area_km2 = PIXEL_SIZE_M ** 2 / 1e6
    counts = pd.Series(class_map.ravel()).value_counts().sort_index()
    area = pd.DataFrame({"class": [RISK_LABELS[i] for i in counts.index],
                         "pixels": counts.values,
                         "area_km2": (counts.values * area_km2).round(3)})
    print(area.to_string(index=False))
    area.to_csv(OUT / "step6_class_area.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    im = ax[0].imshow(prob_map, cmap="magma", vmin=0, vmax=1)
    ax[0].set_title("Flood probability")
    plt.colorbar(im, ax=ax[0], fraction=0.046)
    cmap = ListedColormap(["#2ecc71", "#f1c40f", "#e67e22", "#c0392b"])
    ax[1].imshow(class_map, cmap=cmap, vmin=0, vmax=3)
    ax[1].set_title("Risk classes: " + " / ".join(RISK_LABELS))
    for a in ax:
        a.axis("off")
    fig.text(0.5, 0.01, "Research / demo output - not an operational emergency warning",
             ha="center", fontsize=8, color="gray")
    plt.tight_layout()
    plt.savefig(OUT / "step6_risk_map.png", dpi=110)


if __name__ == "__main__":
    main()
