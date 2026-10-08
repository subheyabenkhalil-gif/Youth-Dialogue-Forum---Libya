"""
config.py
الإعدادات المشتركة لكل الخطوات (المسارات، الطبقات، فئات الخطر).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"            # الطبقات الأصلية (npy أو tif)
PROC = ROOT / "data" / "processed"     # الطبقات بعد التنظيف + جدول الخصائص
OUT = ROOT / "outputs"                 # الأشكال والنتائج
for _p in (RAW, PROC, OUT):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42
GRID = (200, 200)       # حجم الشبكة في البيانات التجريبية فقط
PIXEL_SIZE_M = 30.0     # حجم البكسل بالمتر (لحساب الانحدار) - عدّله حسب بياناتك

# أسماء ملفات الطبقات داخل data/raw  (name.npy أو name.tif)
LAYERS = [
    "dem",            # نموذج الارتفاعات الرقمي (م)
    "rain",           # مجموع الأمطار (مم)
    "wadi_mask",      # شبكة الأودية (1 = مجرى، 0 = غير ذلك)
    "urban",          # المناطق العمرانية (1/0)
    "s1_vv_pre",      # Sentinel-1 VV قبل الحدث (dB)
    "s2_green", "s2_red", "s2_nir", "s2_swir",   # Sentinel-2
    "flood_ref",      # الغمر التاريخي: مرجع للتحقق فقط - لا يدخل كخاصية
]
# طبقات اختيارية
OPTIONAL = [
    "s1_vv_post",     # Sentinel-1 أثناء/بعد الحدث (dB)
    "hyper813",       # بيانات 813 فائقة الطيف: شكلها (bands, H, W) - عند توفرها فقط
]

# تحذير منهجي: إذا كان مرجع الغمر مشتقًا من SAR أثناء الحدث،
# فاستخدام SAR نفسه كخاصية يجعل التقييم دائريًا (circular). لذلك الافتراضي False.
USE_EVENT_SAR_AS_FEATURE = False

# تحويل الاحتمال إلى فئات خطر
RISK_BINS = [0.25, 0.50, 0.75]
RISK_LABELS = ["Low", "Medium", "High", "Severe"]

# الفصل المكاني: حجم الكتلة بالبكسل، ونسبة الكتل المحجوزة للاختبار
BLOCK = 40
TEST_FRACTION = 0.30
