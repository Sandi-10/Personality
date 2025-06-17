elif page == "Anggota Kelompok":
    st.title("👥 Anggota Kelompok")
    
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/Student_icon.svg/800px-Student_icon.svg.png", width=150, caption="Kelompok Analisis Kepribadian")

    st.markdown("### 👩‍🎓👨‍🎓 Daftar Anggota:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("https://i.pravatar.cc/100?img=5", width=100)
        st.markdown("**1. Diva Auliya Pusparini**  
        NIM: `2304030041`")

        st.image("https://i.pravatar.cc/100?img=20", width=100)
        st.markdown("**2. Paskalia Kanicha Mardian**  
        NIM: `2304030062`")

    with col2:
        st.image("https://i.pravatar.cc/100?img=15", width=100)
        st.markdown("**3. Sandi Krisna Mukti**  
        NIM: `2304030074`")

        st.image("https://i.pravatar.cc/100?img=8", width=100)
        st.markdown("**4. Siti Maisyaroh**  
        NIM: `2304030079`")

    st.markdown("---")
    st.success("💡 Aplikasi ini dibuat untuk mendukung pembelajaran tentang klasifikasi tipe kepribadian berdasarkan data!")
