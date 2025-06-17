import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ================== CUSTOM CSS ==================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
        background-image: linear-gradient(to bottom right, #f0f2f6, #dbe9f4);
    }
    h1, h2, h3, h4, h5, h6, p {
        color: #333333;
    }
    .css-1d391kg {
        background-color: #e0e0e0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================== LOAD DATA ==================
url = 'https://raw.githubusercontent.com/Sandi-10/Personality/main/personality_dataset.csv'
df = pd.read_csv(url)

# LabelEncoder untuk target
target_encoder = LabelEncoder()
df['Personality'] = target_encoder.fit_transform(df['Personality'])

# ================== SIDEBAR ==================
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Informasi", "Pemodelan Data", "Prediksi", "Anggota Kelompok"])

# ================== INFORMASI ==================
if page == "Informasi":
    st.title("Informasi Dataset")
    st.write("Dataset ini berisi data kepribadian berdasarkan berbagai aspek.")
    st.subheader("Contoh Data")
    st.dataframe(df.head())

    st.subheader("Deskripsi Kolom")
    st.write(df.describe(include='all'))

    st.subheader("Distribusi Target (Personality Type)")
    st.bar_chart(df['Personality'].value_counts())

    st.subheader("Heatmap Korelasi Fitur")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ================== PEMODELAN ==================
elif page == "Pemodelan Data":
    st.title("Pemodelan Data")

    df_model = df.copy()
    X = df_model.drop('Personality', axis=1)
    y = df_model['Personality']

    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    if st.button("Latih Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.subheader("Akurasi Model")
        st.write(f"Akurasi: {acc:.2f}")

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

        # Visualisasi pentingnya fitur
        st.subheader("Visualisasi Pentingnya Fitur")
        feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig2, ax2 = plt.subplots()
        sns.barplot(x=feature_importance, y=feature_importance.index, ax=ax2)
        ax2.set_xlabel("Importance")
        ax2.set_ylabel("Features")
        st.pyplot(fig2)

# ================== PREDIKSI ==================
elif page == "Prediksi":
    st.title("Prediksi Kepribadian")
    st.write("Masukkan nilai fitur untuk memprediksi tipe kepribadian:")

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

        model = RandomForestClassifier()
        X_all = df.drop('Personality', axis=1)
        y_all = df['Personality']

        for col in X_all.columns:
            if X_all[col].dtype == 'object':
                le = LabelEncoder()
                X_all[col] = le.fit_transform(X_all[co]()_
