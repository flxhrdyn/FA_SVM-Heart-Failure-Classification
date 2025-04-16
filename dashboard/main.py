import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE



# Load model dan scaler
def load_model_and_scaler(model_path="dashboard\fa_svm_model.pkl", scaler_path="dashboard\scaler.pkl"):
    with open(model_path, "rb") as model_file, open(scaler_path, "rb") as scaler_file:
        model = pickle.load(model_file)
        scaler = pickle.load(scaler_file)
    return model, scaler

saved_final_model, saved_scaler = load_model_and_scaler()

# Sidebar
st.sidebar.title("Navigasi Dashboard")
page = st.sidebar.radio("Pilih Halaman:", ["Dashboard", "Prediksi Pasien Baru"])

# ======================================
# DASHBOARD
# ======================================
if page == "Dashboard":
    st.title("📉Heart Failure Classification")
    
    uploaded_file = st.file_uploader("📂 Upload Dataset CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("🔍 Data Preview")
        st.write(df.head())
        
        with st.expander("📊 Descriptive Statistics"):
            st.write(df.describe())
        
        with st.expander("📈 Correlation Heatmap"):
            plt.figure(figsize=(10, 6))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            st.pyplot(plt)
        
        X = df.drop(columns=["DEATH_EVENT"])
        y = df["DEATH_EVENT"]

        # Normalize
        X_scaled = saved_scaler.transform(X)

        # Balance with SMOTE
        smote = SMOTE(sampling_strategy=1.0, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Predict with resampled data
        y_pred = saved_final_model.predict(X_resampled)

        # Evaluation
        final_accuracy = accuracy_score(y_resampled, y_pred)
        st.subheader("📈 Model Evaluation")
        st.write(f"**Akurasi Model:** {final_accuracy:.4f}")
        
        with st.expander("📑 Classification Report"):
            report_dict = classification_report(
                y_resampled, y_pred, target_names=["Selamat", "Meninggal"], output_dict=True
            )
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

        
        with st.expander("📊 Confusion Matrix"):
            cm = confusion_matrix(y_resampled, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Selamat", "Meninggal"],
                        yticklabels=["Selamat", "Meninggal"])
            plt.xlabel("Prediksi")
            plt.ylabel("Aktual")
            st.pyplot(plt)

# ======================================
# PREDIKSI PASIEN BARU
# ======================================
elif page == "Prediksi Pasien Baru":
    st.title("🧑‍⚕️ Prediksi Data Pasien Baru")
    
    with st.form("patient_form"):
        st.subheader("Masukkan Data Pasien")

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Usia", min_value=0, value=60)
            anaemia = st.radio("Anaemia", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
            creatinine_phosphokinase = st.number_input("CPK (mcg/L)", min_value=0, value=250)
            diabetes = st.radio("Diabetes", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
            ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100, value=40)
            high_blood_pressure = st.radio("Hipertensi", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
        
        with col2:
            platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=0.0, value=250000.0)
            serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, value=1.1)
            serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=0, value=137)
            sex = st.radio("Jenis Kelamin", [0, 1], format_func=lambda x: "Wanita" if x == 0 else "Pria")
            smoking = st.radio("Merokok", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
            time = st.number_input("Follow-up Time (days)", min_value=0, value=100)

        submitted = st.form_submit_button("Prediksi")

    if submitted:
        # Sesuaikan urutan fitur dengan dataset pelatihan
        input_dict = {
            'age': age,
            'anaemia': anaemia,
            'creatinine_phosphokinase': creatinine_phosphokinase,
            'diabetes': diabetes,
            'ejection_fraction': ejection_fraction,
            'high_blood_pressure': high_blood_pressure,
            'platelets': platelets,
            'serum_creatinine': serum_creatinine,
            'serum_sodium': serum_sodium,
            'sex': sex,
            'smoking': smoking,
            'time': time
        }

        input_df = pd.DataFrame([input_dict])
        input_scaled = saved_scaler.transform(input_df)
        prediction = saved_final_model.predict(input_scaled)

        result = "Meninggal" if prediction[0] == 1 else "Selamat"
        if result == "Meninggal":
            st.warning(f"### Hasil Prediksi: {result}")
        else:
            st.success(f"### Hasil Prediksi: {result}")

