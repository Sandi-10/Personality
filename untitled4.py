import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load data
url = 'https://raw.githubusercontent.com/Sandi-10/Personality/main/personality_dataset.csv'
df = pd.read_csv(url)

# LabelEncoder untuk target
target_encoder = LabelEncoder()
df['Personality'] = target_encoder.fit_transform(df['Personality'])

# Inisialisasi session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = None

# Sidebar navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Informasi", "Pemodelan Data", "Prediksi"])

# Halaman Informasi
if page == "Informasi":
    st.title("Informasi Dataset")
    st.write("Dataset ini berisi data kepribadian berdasarkan berbagai aspek.")
    st.subheader("Contoh Data")
    st.dataframe(df.head())

    st.subheader("Deskripsi Kolom")
    st.write(df.describe(include='all'))

    st.subheader("Distribusi Target (Personality Type)")
    st.bar_chart(df['Personality'].value_counts())

# Halaman Pemodelan
elif page == "Pemodelan Data":
    st.title("Pemodelan Data")

    df_model = df.copy()
    X = df_model.drop('Personality', axis=1)
    y = df_model['Personality']

    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if st.button("Latih Model"):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.subheader("Akurasi Model")
        st.write(f"Akurasi: {acc:.2f}")

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

        # Simpan model & kolom fitur ke session state
        st.session_state.model = model
        st.session_state.X_columns = X.columns.tolist()
    else:
        st.info("Klik tombol 'Latih Model' untuk memulai pelatihan.")

# Halaman Prediksi
elif page == "Prediksi":
    st.title("Prediksi Kepribadian")
    st.write("Masukkan nilai fitur untuk memprediksi tipe kepribadian:")

    if st.session_state.model is None:
        st.warning("Model belum dilatih. Silakan buka halaman 'Pemodelan Data' dan latih model terlebih dahulu.")
    else:
        input_data = {}
        for col in df.columns:
            if col != 'Personality':
                if df[col].dtype in [np.float64, np.int64]:
                    val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
                else:
                    val = st.selectbox(f"{col}", sorted(df[col].dropna().unique()))
                input_data[col] = val

        if st.button("Prediksi"):
            input_df = pd.DataFrame([input_data])

            for col in input_df.columns:
                if input_df[col].dtype == 'object':
                    le = LabelEncoder()
                    le.fit(df[col])
                    input_df[col] = le.transform(input_df[col])

            input_df = input_df[st.session_state.X_columns]
            prediction = st.session_state.model.predict(input_df)[0]
            predicted_label = target_encoder.inverse_transform([prediction])[0]
            st.success(f"Tipe Kepribadian yang Diprediksi: {predicted_label}")
