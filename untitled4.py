import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
import io

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

# Fungsi konversi gambar ke base64
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Ganti path sesuai dengan file yang diupload
bg_image = get_base64("/mnt/data/024692ef-696e-49aa-9869-483952a3e36c.jpg")
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

# Load dataset
url = 'https://raw.githubusercontent.com/Sandi-10/Personality/main/personality_dataset.csv'
df = pd.read_csv(url)

# Encode target
target_encoder = LabelEncoder()
df['Personality'] = target_encoder.fit_transform(df['Personality'])

# Inisialisasi Session State
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None

# Sidebar
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Informasi", "Pemodelan Data", "Prediksi", "Tuning Model", "Anggota Kelompok"])

# ------------------ Informasi Dataset ------------------
if page == "Informasi":
    st.title("📘 Informasi Dataset")
    st.write("Dataset ini berisi data kepribadian berdasarkan berbagai aspek.")
    st.dataframe(df.head())
    st.subheader("Distribusi Target")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Personality', data=df, ax=ax1)
    ax1.set_xticklabels(target_encoder.inverse_transform(sorted(df['Personality'].unique())))
    st.pyplot(fig1)

# ------------------ Pemodelan Data ------------------
elif page == "Pemodelan Data":
    st.title("📊 Pemodelan Data")

    df_model = df.copy()
    X = df_model.drop('Personality', axis=1)
    y = df_model['Personality']

    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_type = st.selectbox("Pilih Model", ["Random Forest", "KNN", "SVM"])

    if st.button("Latih Model"):
        if model_type == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif model_type == "KNN":
            model = KNeighborsClassifier()
        else:
            model = SVC(probability=True)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.session_state.model = model
        st.session_state.X_columns = X.columns.tolist()
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred

        acc = accuracy_score(y_test, y_pred)
        st.metric("Akurasi", f"{acc:.2f}")
        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# ------------------ Prediksi ------------------
elif page == "Prediksi":
    st.title("🔮 Prediksi Kepribadian")

    if st.session_state.model is None:
        st.warning("Latih model terlebih dahulu di halaman Pemodelan Data.")
    else:
        input_data = {}
        for col in df.columns:
            if col != 'Personality':
                if df[col].dtype in [np.int64, np.float64]:
                    input_data[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
                else:
                    input_data[col] = st.selectbox(f"{col}", sorted(df[col].dropna().unique()))

        input_df = pd.DataFrame([input_data])
        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                le = LabelEncoder()
                le.fit(df[col])
                input_df[col] = le.transform(input_df[col])

        input_df = input_df[st.session_state.X_columns]

        if st.button("Prediksi"):
            prediction = st.session_state.model.predict(input_df)[0]
            predicted_label = target_encoder.inverse_transform([prediction])[0]
            st.success(f"Tipe kepribadian yang diprediksi: {predicted_label}")

            # Unduh hasil
            output = pd.DataFrame({"Prediksi": [predicted_label]})
            csv = output.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Unduh Hasil Prediksi", csv, "hasil_prediksi.csv", "text/csv")

# ------------------ Tuning Model ------------------
elif page == "Tuning Model":
    st.title("🛠️ Tuning Hyperparameter")

    df_model = df.copy()
    X = df_model.drop('Personality', axis=1)
    y = df_model['Personality']

    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_option = st.selectbox("Pilih Model untuk Tuning", ["Random Forest", "KNN"])

    if model_option == "Random Forest":
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 5, 10]
        }
        model = RandomForestClassifier(random_state=42)
    else:
        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
        model = KNeighborsClassifier()

    if st.button("🔍 Mulai Tuning"):
        grid = GridSearchCV(model, param_grid, cv=3)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        acc = accuracy_score(y_test, best_model.predict(X_test))

        st.success(f"Akurasi terbaik: {acc:.2f}")
        st.json(grid.best_params_)

# ------------------ Anggota Kelompok ------------------
elif page == "Anggota Kelompok":
    st.title("👥 Anggota Kelompok")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("/mnt/data/024692ef-696e-49aa-9869-483952a3e36c.jpg", width=180)
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
