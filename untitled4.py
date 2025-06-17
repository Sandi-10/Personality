import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

# Fungsi untuk mengubah nama kolom menjadi format label yang mudah dibaca
def format_label(label):
    return label.replace('_', ' ').title().replace('Dan', 'dan')

# Load data
url = 'https://raw.githubusercontent.com/Sandi-10/Personality/main/personality_dataset.csv'
df = pd.read_csv(url)

# Encode target
target_encoder = LabelEncoder()
df['Tipe_Kepribadian'] = target_encoder.fit_transform(df['Tipe_Kepribadian'])

# Inisialisasi session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# Sidebar navigasi
st.sidebar.title("Navigasi")
halaman = st.sidebar.radio("Pilih Halaman:", ["Informasi", "Pemodelan Data", "Prediksi", "Anggota Kelompok"])

# -------------------- Halaman Informasi --------------------
if halaman == "Informasi":
    st.title("📘 Informasi Dataset")
    st.write("Dataset ini berisi data kepribadian berdasarkan berbagai aspek.")

    st.subheader("🔍 Contoh Data")
    st.dataframe(df.head())

    st.subheader("📊 Deskripsi Kolom")
    st.write(df.describe(include='all'))

    st.subheader("🧠 Distribusi Target (Tipe Kepribadian)")
    fig_dist, ax_dist = plt.subplots()
    sns.countplot(data=df, x='Tipe_Kepribadian', ax=ax_dist)
    ax_dist.set_xticklabels(target_encoder.inverse_transform(sorted(df['Tipe_Kepribadian'].unique())))
    st.pyplot(fig_dist)

    st.subheader("📉 Korelasi antar Fitur")
    fig_corr, ax_corr = plt.subplots()
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

    st.subheader("📦 Boxplot Setiap Fitur Numerik")
    for kolom in df.select_dtypes(include=['int64', 'float64']).columns:
        if kolom != 'Tipe_Kepribadian':
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='Tipe_Kepribadian', y=kolom, ax=ax)
            ax.set_title(f"Distribusi {format_label(kolom)} berdasarkan Tipe Kepribadian")
            ax.set_xticklabels(target_encoder.inverse_transform(sorted(df['Tipe_Kepribadian'].unique())))
            st.pyplot(fig)

# -------------------- Halaman Pemodelan --------------------
elif halaman == "Pemodelan Data":
    st.title("📊 Pemodelan Data")

    df_model = df.copy()
    X = df_model.drop('Tipe_Kepribadian', axis=1)
    y = df_model['Tipe_Kepribadian']

    # Encode fitur kategorikal
    for kolom in X.columns:
        if X[kolom].dtype == 'object':
            le = LabelEncoder()
            X[kolom] = le.fit_transform(X[kolom])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if st.button("🚀 Latih Model"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        akurasi = accuracy_score(y_test, y_pred)
        laporan = classification_report(y_test, y_pred, target_names=target_encoder.classes_, output_dict=True)

        st.session_state.model = model
        st.session_state.X_columns = X.columns.tolist()
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        st.subheader("🎯 Akurasi Model")
        st.metric(label="Akurasi", value=f"{akurasi:.2f}")

        st.subheader("📋 Laporan Klasifikasi")
        laporan_df = pd.DataFrame(laporan).transpose()
        st.dataframe(laporan_df.style.format("{:.2f}"))

        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_encoder.classes_,
                    yticklabels=target_encoder.classes_, ax=ax)
        ax.set_xlabel('Prediksi')
        ax.set_ylabel('Aktual')
        st.pyplot(fig_cm)

        st.subheader("📌 Pentingnya Fitur")
        penting = model.feature_importances_
        df_penting = pd.DataFrame({'Fitur': X.columns, 'Pentingnya': penting}).sort_values(by='Pentingnya', ascending=False)
        fig_imp, ax2 = plt.subplots()
        sns.barplot(x='Pentingnya', y='Fitur', data=df_penting, palette='viridis', ax=ax2)
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
elif halaman == "Prediksi":
    st.title("🔮 Prediksi Tipe Kepribadian")
    st.write("Masukkan nilai fitur untuk memprediksi tipe kepribadian:")

    if st.session_state.model is None:
        st.warning("Model belum dilatih. Silakan buka halaman 'Pemodelan Data' dan klik tombol 'Latih Model'.")
    else:
        input_data = {}
        for kolom in df.columns:
            if kolom != 'Tipe_Kepribadian':
                label_tampil = format_label(kolom)
                if df[kolom].dtype in [np.float64, np.int64]:
                    val = st.number_input(label_tampil, float(df[kolom].min()), float(df[kolom].max()), float(df[kolom].mean()))
                else:
                    val = st.selectbox(label_tampil, sorted(df[kolom].dropna().unique()))
                input_data[kolom] = val

        input_df = pd.DataFrame([input_data])

        if st.button("Prediksi"):
            for kolom in input_df.columns:
                if input_df[kolom].dtype == 'object':
                    le = LabelEncoder()
                    le.fit(df[kolom])
                    input_df[kolom] = le.transform(input_df[kolom])

            input_df = input_df[st.session_state.X_columns]
            prediksi = st.session_state.model.predict(input_df)[0]
            probabilitas = st.session_state.model.predict_proba(input_df)[0]
            label_prediksi = target_encoder.inverse_transform([prediksi])[0]

            st.success(f"✅ Tipe Kepribadian yang Diprediksi: **{label_prediksi}**")

            st.subheader("📋 Input Anda")
            st.dataframe(input_df)

            st.subheader("📈 Probabilitas Prediksi")
            prob_df = pd.Series(probabilitas, index=target_encoder.classes_)
            st.bar_chart(prob_df)

# -------------------- Halaman Anggota Kelompok --------------------
elif halaman == "Anggota Kelompok":
    st.title("👥 Anggota Kelompok")

    st.markdown("""
    ### 👩‍🏫 **Diva Auliya Pusparini**  
    🆔 NIM: 2304030041  

    ### 👩‍🎓 **Paskalia Kanicha Mardian**  
    🆔 NIM: 2304030062  

    ### 👨‍💻 **Sandi Krisna Mukti**  
    🆔 NIM: 2304030074  

    ### 👩‍⚕️ **Siti Maisyaroh**  
    🆔 NIM: 2304030079
    """)
