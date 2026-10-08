# 813 DERNA – خط تحليل البيانات (خطوة بخطوة)

كل خطوة في ملف مستقل، ويمكن تشغيلها منفردة أو كلها معًا.

## التشغيل السريع (بيانات تجريبية للتأكد أن الكود يعمل)
```bash
pip install -r requirements.txt
python run_all.py --demo                 # بدون 813
python run_all.py --demo --with-hyper    # مع اختبار خطوة 813
```
> البيانات التجريبية **Synthetic** وليست بيانات درنة. أي أرقام تخرج منها ليست نتائج علمية.

## التشغيل على بياناتك الحقيقية
1. جهّز الطبقات بنفس الإسقاط والدقة والحدود (في QGIS/GDAL/rasterio).
2. ضعها في `data/raw/` بأسماء `config.LAYERS` (صيغة `.npy` أو `.tif`).
3. شغّل الخطوات من 1 إلى 8 (أو `python run_all.py`).

## الخطوات
| الملف | الوظيفة | المخرجات |
|---|---|---|
| `step0_make_demo_data.py` | توليد بيانات تجريبية (تُتجاوز مع البيانات الحقيقية) | `data/raw/*.npy` |
| `step1_load_inspect.py` | تحميل الطبقات، فحص المحاذاة، إحصاءات، نسبة القيم المفقودة | `step1_summary.csv`, `step1_quicklook.png` |
| `step2_preprocess.py` | ملء الفجوات (سحب)، قص القيم الشاذة | `data/processed/clean_layers.npz` |
| `step3_features.py` | NDVI, NDWI, الانحدار, المسافة للأودية, الأمطار... | `data/processed/features.csv` |
| `step4_eda.py` | توازن الفئات، AUC لكل خاصية، الارتباط، Boxplots | `step4_*` |
| `step5_train_baseline.py` | Random Forest + فصل مكاني + تحقق متقاطع مكاني | `baseline_model.joblib` |
| `step6_risk_map.py` | خريطة الاحتمالية وفئات الخطر الأربع | `step6_risk_map.png` |
| `step7_validate.py` | Precision / Recall / F1 / ROC-AUC / مصفوفة الالتباس | `step7_metrics.json` |
| `step8_compare_813.py` | مقارنة Baseline مع Enhanced (خصائص 813 عبر PCA) | `step8_comparison.csv` |

## ملاحظات منهجية مهمة
- **طبقة الغمر التاريخي** مرجع للتحقق فقط، ولا تدخل كخاصية.
- **الدائرية (Circularity):** إذا كان مرجع الغمر مشتقًا من SAR أثناء الحدث، فلا تستخدم SAR الحدث كخاصية
  (`USE_EVENT_SAR_AS_FEATURE=False` افتراضيًا في `config.py`).
- **التسرب المكاني:** الفصل بين التدريب والاختبار بالكتل المكانية وليس بالبكسل العشوائي.
- **خطوة 813:** تُنفَّذ فقط عند توفر بيانات مصرح بها، وتُقارَن بالـ Baseline بدل افتراض التحسن.
  كرّر المقارنة بعدة بذور وتقسيمات قبل أي ادعاء.
- الخريطة الناتجة نموذج بحثي/إثبات مفهوم وليست نظام إنذار طوارئ معتمدًا.
