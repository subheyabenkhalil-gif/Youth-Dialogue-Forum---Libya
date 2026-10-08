"""
الخطوة 4: التحليل الاستكشافي للبيانات (EDA).
-------------------------------------------------
ماذا نفعل؟
  - توازن الفئات (غمر / لا غمر).
  - مقارنة متوسط كل خاصية بين المناطق المغمورة وغير المغمورة.
  - ROC-AUC لكل خاصية منفردة (مؤشر مبدئي على قوة الفصل).
  - مصفوفة الارتباط بين الخصائص (لكشف التكرار).
المخرجات: outputs/step4_*.csv و outputs/step4_*.png
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from config import PROC, OUT

META = ["row", "col", "flood"]


def main():
    df = pd.read_csv(PROC / "features.csv")
    feats = [c for c in df.columns if c not in META]

    print("توازن الفئات:\n", df["flood"].value_counts(normalize=True).round(3).to_string())

    means = df.groupby("flood")[feats].mean().T.round(3)
    means.columns = ["no_flood_mean", "flood_mean"]
    print("\nمتوسط الخصائص حسب الفئة:\n", means.to_string())
    means.to_csv(OUT / "step4_class_means.csv")

    auc = {f: roc_auc_score(df["flood"], df[f]) for f in feats}
    auc = pd.Series(auc).rename("univariate_auc")
    # AUC < 0.5 تعني أن العلاقة عكسية؛ نعرض القوة بغض النظر عن الاتجاه
    auc_df = pd.DataFrame({"auc": auc, "strength": (auc - 0.5).abs() + 0.5}).sort_values("strength", ascending=False)
    print("\nقوة الفصل لكل خاصية:\n", auc_df.round(3).to_string())
    auc_df.to_csv(OUT / "step4_univariate_auc.csv")

    corr = df[feats].corr()
    corr.round(3).to_csv(OUT / "step4_correlation.csv")
    high = [(a, b, round(corr.loc[a, b], 2)) for i, a in enumerate(feats)
            for b in feats[i + 1:] if abs(corr.loc[a, b]) > 0.85]
    print("\nأزواج شديدة الارتباط (>0.85):", high or "لا يوجد")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats)), feats, rotation=60, ha="right")
    ax.set_yticks(range(len(feats)), feats)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(OUT / "step4_correlation.png", dpi=110)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, f in zip(axes.ravel(), feats):
        df.boxplot(column=f, by="flood", ax=ax, showfliers=False)
        ax.set_title(f, fontsize=9)
        ax.set_xlabel("")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(OUT / "step4_boxplots.png", dpi=110)
    print("تم الحفظ في outputs/")


if __name__ == "__main__":
    main()
