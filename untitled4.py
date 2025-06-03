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

# Coba ubah kolom numerik jika perlu (karena koma)
for col in df.columns:
    if col != 'Personality':
        try:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        except:
            pass

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

# Halaman Pemodelan Data
elif page == "Pemodelan Data":
    st.title("Pemodelan Data")

    df_model = df.copy()
    label_encoder = LabelEncoder()
    df_model['Personality'] = label_encoder.fit_transform(df_model['Personality'])

    X = df_model.drop('Personality', axis=1)
    y = df_model['Personality']

    # Encoding untuk kolom non-numerik
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.subheader("Akurasi Model")
    st.write(f"Akurasi: {acc:.2f}")

    st.subheader("Classification Report")
    unique_labels = np.unique(y_test)
    target_names = label_encoder.inverse_transform(unique_labels)
    st.text(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))

# Halaman Prediksi
elif page == "Prediksi":
    st.title("Prediksi Kepribadian")
    st.write("Masukkan nilai fitur untuk memprediksi tipe kepribadian:")

    input_data = {}
    for col in df.columns:
        if col != 'Personality':
            if pd.api.types.is_numeric_dtype(df[col]):
                val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            else:
                unique_vals = df[col].dropna().unique()
                val = st.selectbox(f"{col}", sorted(unique_vals.astype(str)))
            input_data[col] = val

    if st.button("Prediksi"):
        input_df = pd.DataFrame([input_data])

        # Encoding input
        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                le_input = LabelEncoder()
                le_input.fit(df[col].astype(str))
                input_df[col] = le_input.transform(input_df[col].astype(str))

        # Label target
        label_encoder = LabelEncoder()
        label_encoder.fit(df['Personality'])

        # Gunakan model dari halaman pemodelan
        model = RandomForestClassifier()
        df_model = df.copy()
        df_model['Personality'] = label_encoder.fit_transform(df_model['Personality'])

        X = df_model.drop('Personality', axis=1)
        y = df_model['Personality']
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        model.fit(X, y)
        prediction = model.predict(input_df)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]

        st.success(f"Tipe Kepribadian yang Diprediksi: {predicted_label}")
