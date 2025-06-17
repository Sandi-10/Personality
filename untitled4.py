import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

# Fungsi untuk mengonversi gambar ke base64
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Tambahkan background gambar ke seluruh halaman
bg_image = get_base64("a14f21d8-501c-4e9f-86d7-79e649c615c8.jpg")
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load data
url = 'https://raw.githubusercontent.com/Sandi-10/Personality/main/personality_dataset.csv'
df = pd.read_csv(url)

# Encode target
target_encoder = LabelEncoder()
df['Personality'] = target_encoder.fit_transform(df['Personality'])

# Inisialisasi session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = 'Random Forest'

# Sidebar navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Informasi", "Pemodelan Data", "Prediksi", "Anggota Kelompok"])

# -------------------- Halaman Informasi --------------------
if page == "Informasi":
    st.title("📘 Informasi Dataset")
    st.write("Dataset ini berisi data kepribadian berdasarkan berbagai aspek.")

    st.subheader("🔍 Contoh Data")
    st.dataframe(df.head())

    st.subheader("📊 Deskripsi Kolom")
    st.write(df.describe(include='all'))

    st.subheader("🧠 Distribusi Target (Personality Type)")
    fig_dist, ax_dist = plt.subplots()
    sns.countplot(data=df, x='Personality', ax=ax_dist)
    ax_dist.set_xticklabels(target_encoder.inverse_transform(sorted(df['Personality'].unique())))
    st.pyplot(fig_dist)

    st.subheader("📉 Korelasi antar Fitur")
    fig_corr, ax_corr = plt.subplots()
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

    st.subheader("📦 Boxplot Setiap Fitur Numerik")
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if col != 'Personality':
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='Personality', y=col, ax=ax)
            ax.set_title(f"Distribusi {col} berdasarkan Personality")
            ax.set_xticklabels(target_encoder.inverse_transform(sorted(df['Personality'].unique())))
            st.pyplot(fig)

# -------------------- Halaman Pemodelan --------------------
elif page == "Pemodelan Data":
    st.title("📊 Pemodelan Data")

    df_model = df.copy()
    X = df_model.drop('Personality', axis=1)
    y = df_model['Personality']

    # Encode fitur kategorikal
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.subheader("🔧 Pilih Algoritma")
    model_type = st.selectbox("Pilih Model:", ["Random Forest", "K-Nearest Neighbors", "Support Vector Machine"])

    if st.button("🚀 Latih Model"):
        if model_type == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif model_type == "K-Nearest Neighbors":
            model = KNeighborsClassifier()
        elif model_type == "Support Vector Machine":
            model = SVC(probability=True)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_encoder.classes_, output_dict=True)

        st.session_state.model = model
        st.session_state.X_columns = X.columns.tolist()
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.model_type = model_type

        st.subheader("🎯 Akurasi Model")
        st.metric(label="Akurasi", value=f"{acc:.2f}")

        st.subheader("📋 Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))

        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_encoder.classes_,
                    yticklabels=target_encoder.classes_, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig_cm)

        if hasattr(model, 'feature_importances_'):
            st.subheader("📌 Pentingnya Fitur")
            importances = model.feature_importances_
            imp_df = pd.DataFrame({'Fitur': X.columns, 'Pentingnya': importances}).sort_values(by='Pentingnya', ascending=False)
            fig_imp, ax2 = plt.subplots()
            sns.barplot(x='Pentingnya', y='Fitur', data=imp_df, palette='viridis', ax=ax2)
            ax2.set_title("Pentingnya Fitur")
            st.pyplot(fig_imp)

        if len(target_encoder.classes_) == 2:
            st.subheader("🚦 ROC Curve")
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig_roc, ax3 = plt.subplots()
            ax3.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax3.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax3.set_title("ROC Curve")
            ax3.set_xlabel("False Positive Rate")
            ax3.set_ylabel("True Positive Rate")
            ax3.legend()
            st.pyplot(fig_roc)

# -------------------- Halaman Prediksi --------------------
elif page == "Prediksi":
    st.title("🔮 Prediksi Kepribadian")
    st.write("Masukkan nilai fitur untuk memprediksi tipe kepribadian:")

    if st.session_state.model is None:
        st.warning("Model belum dilatih. Silakan buka halaman 'Pemodelan Data' dan klik tombol 'Latih Model'.")
    else:
        input_data = {}
        for col in df.columns:
            if col != 'Personality':
                if df[col].dtype in [np.float64, np.int64]:
                    val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
                else:
                    val = st.selectbox(f"{col}", sorted(df[col].dropna().unique()))
                input_data[col] = val

        input_df = pd.DataFrame([input_data])

        if st.button("Prediksi"):
            for col in input_df.columns:
                if input_df[col].dtype == 'object':
                    le = LabelEncoder()
                    le.fit(df[col])
                    input_df[col] = le.transform(input_df[col])

            input_df = input_df[st.session_state.X_columns]
            prediction = st.session_state.model.predict(input_df)[0]
            prob = st.session_state.model.predict_proba(input_df)[0]
            predicted_label = target_encoder.inverse_transform([prediction])[0]

            st.success(f"✅ Tipe Kepribadian yang Diprediksi: *{predicted_label}*")

            st.subheader("📋 Input Anda")
            st.dataframe(input_df)

            st.subheader("📈 Probabilitas Prediksi")
            prob_df = pd.Series(prob, index=target_encoder.classes_)
            st.bar_chart(prob_df)

            csv = input_df.copy()
            csv['Prediksi'] = predicted_label
            csv = csv.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Unduh Hasil Prediksi", data=csv, file_name="hasil_prediksi.csv", mime="text/csv")

# -------------------- Halaman Anggota Kelompok --------------------
elif page == "Anggota Kelompok":
    st.title("👥 Anggota Kelompok")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("a14f21d8-501c-4e9f-86d7-79e649c615c8.jpg", width=180)
    with col2:
        st.markdown("""
        ### 👩‍🏫 *Diva Auliya Pusparini*  
        🆔 NIM: 2304030041  

        ### 👩‍🎓 *Paskalia Kanicha Mardian*  
        🆔 NIM: 2304030062  

        ### 👨‍💻 *Sandi Krisna Mukti*  
        🆔 NIM: 2304030074  

        ### 👩‍⚕ *Siti Maisyaroh*  
        🆔 NIM: 2304030079
        """)
