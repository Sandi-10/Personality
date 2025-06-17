import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import base64

# URL gambar latar belakang (gunakan gambar sendiri jika perlu)
bg_url = "https://i.imgur.com/yoursampleimage.png"  # Ganti URL ini sesuai gambar kamu

# Tambahkan background
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{bg_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Judul Aplikasi
st.title("Dashboard Prediksi dengan ML")

# ⛳ Tab Navigasi
tab1, tab2, tab3, tab4 = st.tabs(["📊 Eksplorasi Data", "🤖 Pelatihan Model", "📥 Prediksi Baru", "🔧 Tuning Model"])

# 📊 TAB 1: Eksplorasi Data
with tab1:
    st.subheader("Unggah Dataset CSV")
    file = st.file_uploader("Pilih file CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.dataframe(df.head())
        st.write("Dimensi Data:", df.shape)

        if st.checkbox("Tampilkan Statistik Deskriptif"):
            st.write(df.describe())

        if st.checkbox("Tampilkan Heatmap Korelasi"):
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

# 🤖 TAB 2: Pelatihan Model
with tab2:
    st.subheader("Pilih Fitur dan Target")
    if file is not None:
        all_columns = df.columns.tolist()
        fitur = st.multiselect("Fitur (X)", options=all_columns, default=all_columns[:-1])
        target = st.selectbox("Target (y)", options=all_columns)

        X = df[fitur]
        y = df[target]

        test_size = st.slider("Ukuran Data Test", 0.1, 0.5, 0.2)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        model_choice = st.selectbox("Pilih Model", ["Logistic Regression", "SVM", "KNN"])

        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_choice == "SVM":
            model = SVC()
        else:
            model = KNeighborsClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.write("Akurasi:", acc)
        st.text("Laporan Klasifikasi")
        st.text(classification_report(y_test, y_pred))

        # Simpan model dan hasil prediksi
        output_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        csv = output_df.to_csv(index=False).encode()
        st.download_button("📥 Unduh Hasil Prediksi", csv, "hasil_prediksi.csv", "text/csv")

# 📥 TAB 3: Prediksi Input Baru
with tab3:
    st.subheader("Prediksi Data Baru")
    if file is not None and 'model' in locals():
        input_data = {}
        for col in fitur:
            val = st.number_input(f"Masukkan nilai untuk {col}", value=0.0)
            input_data[col] = val
        input_df = pd.DataFrame([input_data])
        pred = model.predict(input_df)[0]
        st.success(f"Hasil Prediksi: {pred}")

# 🔧 TAB 4: Tuning Hyperparameter
with tab4:
    st.subheader("Tuning Hyperparameter")
    if file is not None:
        if model_choice == "SVM":
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
            grid = GridSearchCV(SVC(), param_grid, cv=3)
            grid.fit(X_train, y_train)
            st.write("Best Parameters:", grid.best_params_)
            st.write("Best Score:", grid.best_score_)
        elif model_choice == "KNN":
            param_grid = {
                'n_neighbors': [3, 5, 7, 9]
            }
            grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3)
            grid.fit(X_train, y_train)
            st.write("Best Parameters:", grid.best_params_)
            st.write("Best Score:", grid.best_score_)
        else:
            st.info("Tuning belum tersedia untuk model ini.")
