"""
الخطوة 1: تحميل الطبقات وفحصها.
-------------------------------------------------
ماذا نفعل؟
  - نقرأ كل الطبقات المعرّفة في config.
  - نتأكد أن جميعها بنفس الأبعاد (محاذاة مكانية).
  - نحسب إحصاءات (min/max/mean) ونسبة القيم المفقودة لكل طبقة.
  - نحفظ جدول ملخص + صورة سريعة للطبقات.
المخرجات: outputs/step1_summary.csv , outputs/step1_quicklook.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import LAYERS, OPTIONAL, OUT
from utils import load_layer


def main():
    rows, data = [], {}
    for name in LAYERS + OPTIONAL:
        arr = load_layer(name, required=name in LAYERS)
        if arr is None:
            print(f"- {name}: غير متوفرة (اختيارية)")
            continue
        data[name] = arr
        rows.append(dict(layer=name, shape=str(arr.shape),
                         min=np.nanmin(arr), max=np.nanmax(arr),
                         mean=np.nanmean(arr), std=np.nanstd(arr),
                         pct_missing=100 * np.isnan(arr).mean()))
    summary = pd.DataFrame(rows).round(3)
    print(summary.to_string(index=False))
    summary.to_csv(OUT / "step1_summary.csv", index=False)

    # فحص المحاذاة: كل الطبقات ثنائية الأبعاد يجب أن تتطابق أبعادها
    shapes = {n: a.shape[-2:] for n, a in data.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"الطبقات غير متحاذية مكانيًا: {shapes}\n"
                         "أعد الإسقاط/القص/إعادة العينة قبل المتابعة (انظر المرحلة 2 في الوثيقة).")
    print("\nجميع الطبقات متحاذية بنفس الأبعاد:", set(shapes.values()).pop())

    # صورة سريعة
    show = [n for n in data if data[n].ndim == 2]
    cols = 4
    fig, axes = plt.subplots(-(-len(show) // cols), cols, figsize=(14, 3.2 * -(-len(show) // cols)))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, n in zip(axes.ravel(), show):
        ax.imshow(data[n], cmap="viridis")
        ax.set_title(n, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "step1_quicklook.png", dpi=110)
    print("تم الحفظ في outputs/")


if __name__ == "__main__":
    main()
