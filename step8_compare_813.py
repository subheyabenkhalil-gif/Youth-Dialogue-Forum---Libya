"""
الخطوة 8: مقارنة Baseline مع Enhanced (إضافة خصائص 813 فائقة الطيف).
-------------------------------------------------
- تعمل فقط إذا وُجد data/raw/hyper813 (.npy أو .tif) بالشكل (bands, H, W).
- نضغط النطاقات بـ PCA إلى بضعة مكوّنات ونضيفها كخصائص.
- نستخدم نفس الفصل المكاني ونفس إعدادات النموذج، ثم نقارن المقاييس.
المبدأ (من الوثيقة): لا نفترض أن 813 تحسّن النتائج؛ نقيس ذلك.
المخرجات: outputs/step8_comparison.csv
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from config import PROC, OUT, SEED
from step7_validate import evaluate
from utils import load_layer

META = ["row", "col", "flood"]
N_PC = 5


def main():
    cube = load_layer("hyper813", required=False)
    if cube is None:
        print("بيانات 813 غير متوفرة - تم تخطي الخطوة 8.\n"
              "ضع data/raw/hyper813.npy (bands, H, W) عند توفر البيانات المصرح بها.")
        return
    cube = np.nan_to_num(cube)
    B, H, W = cube.shape
    flat = cube.reshape(B, -1).T
    flat = (flat - flat.mean(0)) / (flat.std(0) + 1e-9)
    pcs = PCA(n_components=N_PC, random_state=SEED).fit_transform(flat)

    df = pd.read_csv(PROC / "features.csv")
    base = [c for c in df.columns if c not in META]
    for i in range(N_PC):
        df[f"h813_pc{i + 1}"] = pcs[:, i]       # الترتيب مطابق لـ row-major نفسه
    enhanced = base + [f"h813_pc{i + 1}" for i in range(N_PC)]

    sp = np.load(OUT / "split.npz")
    tr, te = sp["train"], sp["test"]
    rows = []
    for name, cols in [("baseline", base), ("enhanced_813", enhanced)]:
        rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                    class_weight="balanced", n_jobs=-1, random_state=SEED)
        rf.fit(df.loc[tr, cols], df.loc[tr, "flood"])
        proba = rf.predict_proba(df.loc[te, cols])[:, 1]
        rows.append({"model": name, **evaluate(df.loc[te, "flood"].to_numpy(), proba)})
    res = pd.DataFrame(rows).round(3)
    print(res.to_string(index=False))
    res.to_csv(OUT / "step8_comparison.csv", index=False)
    print("\nتفسير: الفرق الصغير قد يكون ضمن التذبذب العشوائي؛ كرّر بعدة بذور (seeds) "
          "وعدة تقسيمات مكانية قبل الادعاء بأن 813 تحسّن الأداء.")


if __name__ == "__main__":
    main()
