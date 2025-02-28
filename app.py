from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# ✅ تحميل النموذج
model = joblib.load("best_model_catboost.pkl")  # تأكد من أن الملف موجود في نفس المجلد

# ✅ الميزات المستخدمة في التدريب (يجب أن تطابق ميزات التدريب)
selected_features = [
    "person_gender",
    "previous_loan_defaults_on_file",
    "person_education",
    "cb_person_cred_hist_length",
    "loan_percent_income",
    "person_age",
    "person_income",
    "loan_int_rate",
    "person_home_ownership",
    "loan_intent"
]

# ✅ استقبال البيانات والتنبؤ عبر API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 🔹 استلام البيانات من JSON
        data = request.get_json()

        # 🔹 التأكد من أن جميع الميزات المطلوبة موجودة
        missing_features = [feature for feature in selected_features if feature not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        # 🔹 تحويل البيانات إلى DataFrame بنفس ترتيب التدريب
        input_data = pd.DataFrame([data])[selected_features]

        # 🔹 تشغيل التنبؤ
        prediction = model.predict(input_data)[0]

        # 🔹 تحويل النتيجة إلى نص مقروء
        result = "Approved" if prediction == 1 else "Rejected"

        return jsonify({"loan_status": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ تشغيل التطبيق
if __name__ == "__main__":
    app.run(debug=True)

# {
#     "person_gender": 1,  // ذكر
#     "previous_loan_defaults_on_file": 0,  // لم يتخلف عن السداد
#     "person_education": 3,  // بكالوريوس
#     "cb_person_cred_hist_length": 24,  // لديه تاريخ ائتماني لمدة سنتين
#     "loan_percent_income": 15.2,  // القرض يمثل 15.2% من الدخل السنوي
#     "person_age": 35,  // العمر 35 سنة
#     "person_income": 75000,  // الدخل السنوي 75,000 دولار
#     "loan_int_rate": 3.5,  // معدل الفائدة 3.5%
#     "person_home_ownership": 2,  // يمتلك منزلًا
#     "loan_intent": 0  // القرض لغرض التعليم
# }