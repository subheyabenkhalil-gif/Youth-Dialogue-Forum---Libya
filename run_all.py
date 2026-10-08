"""
run_all.py - تشغيل الخطوات بالترتيب.
    python run_all.py --demo     # يولّد بيانات تجريبية أولًا (Synthetic)
    python run_all.py            # يفترض أن بياناتك الحقيقية في data/raw
    python run_all.py --demo --with-hyper   # يختبر أيضًا الخطوة 8
"""
import sys
import importlib

STEPS = ["step1_load_inspect", "step2_preprocess", "step3_features", "step4_eda",
         "step5_train_baseline", "step6_risk_map", "step7_validate", "step8_compare_813"]

if __name__ == "__main__":
    if "--demo" in sys.argv:
        print("\n===== step0_make_demo_data =====")
        importlib.import_module("step0_make_demo_data").main(with_hyper="--with-hyper" in sys.argv)
    for s in STEPS:
        print(f"\n===== {s} =====")
        importlib.import_module(s).main()
    print("\nانتهى التشغيل. النتائج في مجلد outputs/")
