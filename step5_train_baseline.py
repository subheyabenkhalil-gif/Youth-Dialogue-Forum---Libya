"""
الخطوة 5: تدريب النموذج الأساسي Baseline (Random Forest).
-------------------------------------------------
- الخصائص: Sentinel-1 (قبل الحدث) + Sentinel-2 + DEM + الأمطار + الأودية + العمران.
- فصل مكاني بالكتل بين التدريب والاختبار (وليس عشوائيًا بالبكسل).
- التحقق المتقاطع المكاني GroupKFold لتقدير الثبات.
المخرجات: outputs/baseline_model.joblib, outputs/step5_importance.csv/png,
          outputs/split.npz
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from config import PROC, OUT, SEED, BLOCK, TEST_FRACTION
from utils import spatial_split

META = ["row", "col", "flood"]


def main():
    df = pd.read_csv(PROC / "features.csv")
    feats = [c for c in df.columns if c not in META]
    X, y = df[feats], df["flood"]

    tr, te, groups = spatial_split(df, BLOCK, TEST_FRACTION, SEED)
    print(f"تدريب: {tr.sum()} بكسل | اختبار: {te.sum()} بكسل | الخصائص: {len(feats)}")

    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                class_weight="balanced", n_jobs=-1, random_state=SEED)

    cv = cross_val_score(rf, X[tr], y[tr], groups=groups[tr],
                         cv=GroupKFold(n_splits=5), scoring="roc_auc")
    print(f"ROC-AUC بالتحقق المتقاطع المكاني (على التدريب فقط): {cv.mean():.3f} ± {cv.std():.3f}")

    rf.fit(X[tr], y[tr])
    joblib.dump({"model": rf, "features": feats}, OUT / "baseline_model.joblib")
    np.savez(OUT / "split.npz", train=tr, test=te)

    imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
    imp.rename("importance").to_csv(OUT / "step5_importance.csv")
    print("\nأهمية الخصائص:\n", imp.round(3).to_string())
    imp[::-1].plot.barh(figsize=(6, 4))
    plt.title("Feature importance (baseline)")
    plt.tight_layout()
    plt.savefig(OUT / "step5_importance.png", dpi=110)


if __name__ == "__main__":
    main()
