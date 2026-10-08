"""
الخطوة 7: التحقق (Validation) على مناطق الاختبار المحجوزة.
-------------------------------------------------
نحسب: Precision, Recall, F1, ROC-AUC, Average Precision, مصفوفة الالتباس.
المرجع: طبقة الغمر التاريخي (flood_ref) - وهي لم تدخل التدريب كخاصية.
المخرجات: outputs/step7_metrics.json, step7_roc_pr.png, step7_confusion.csv
تنبيه: مع البيانات التجريبية الأرقام ليست نتائج علمية؛ مع بيانات درنة الحقيقية
       تُعتمد النتائج فقط بعد التأكد من جودة المرجع وعدم وجود تسرب.
"""
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
from config import PROC, OUT


def evaluate(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {
        "threshold": thr,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "average_precision": average_precision_score(y, proba),
    }


def main():
    df = pd.read_csv(PROC / "features.csv")
    bundle = joblib.load(OUT / "baseline_model.joblib")
    model, feats = bundle["model"], bundle["features"]
    te = np.load(OUT / "split.npz")["test"]

    y = df.loc[te, "flood"].to_numpy()
    proba = model.predict_proba(df.loc[te, feats])[:, 1]

    m = evaluate(y, proba)
    print(json.dumps({k: round(v, 3) for k, v in m.items()}, indent=2))
    cm = pd.DataFrame(confusion_matrix(y, (proba >= 0.5).astype(int)),
                      index=["actual_no", "actual_flood"], columns=["pred_no", "pred_flood"])
    print("\nمصفوفة الالتباس:\n", cm.to_string())
    cm.to_csv(OUT / "step7_confusion.csv")
    json.dump(m, open(OUT / "step7_metrics.json", "w"), indent=2)

    fpr, tpr, _ = roc_curve(y, proba)
    p, r, _ = precision_recall_curve(y, proba)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr)
    ax[0].plot([0, 1], [0, 1], "--", c="gray")
    ax[0].set_title(f"ROC (AUC={m['roc_auc']:.2f})")
    ax[1].plot(r, p)
    ax[1].set_title(f"Precision-Recall (AP={m['average_precision']:.2f})")
    plt.tight_layout()
    plt.savefig(OUT / "step7_roc_pr.png", dpi=110)


if __name__ == "__main__":
    main()
