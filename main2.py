import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import shap
from sklearn import tree  # untuk tree visualization


# IMPORT DATA
model = joblib.load('./model/lgb_rand.joblib')
encoders = joblib.load('./model/label_encoders.joblib')
explainer = joblib.load('./model/shap_explainer.joblib')


# PAGE CONFIG
def set_page_config():
    st.set_page_config(
        page_title='Credit Risk Scoring',
        page_icon='💳',
        layout='wide',
        initial_sidebar_state='expanded'
    )


# BODY 2 — Risk Scoring (1): SHAP
def body_2():
    st.markdown("<h3>Risk Scoring dengan Interpretasi SHAP</h3>", unsafe_allow_html=True)

    with st.form('Input Data Debitur'):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### Profil Debitur")
            usia = st.number_input('Usia (tahun)', 18, 80, value=21)
            alamat_kota = st.selectbox('Alamat Kota', encoders['alamat_kota'].classes_)
            pendidikan = st.selectbox('Pendidikan', encoders['pendidikan'].keys())
            jenis_usaha = st.selectbox('Jenis Usaha', encoders['jenis_usaha'].classes_)
            gaji = st.select_slider('Penghasilan', options=encoders['gaji(Penghasilan)'].keys(), value='> 7,500,000 S/D 10,000,000')

        with col2:
            st.markdown("#### Riwayat Debitur")
            riwayat_slik = st.selectbox('Riwayat SLIK', encoders['Riwayat Slik'].classes_)
            rek_koran = st.selectbox('Penilaian Rek. Koran', encoders['Penilaian Rekening Koran'].classes_)
            dbr = st.number_input('Debt Burden Ratio (%)', 0., 100., value=50.0, step=0.1)

        with col3:
            st.markdown("#### Ketentuan Pinjaman")
            jenis_pinjaman = st.selectbox('Jenis Pinjaman', encoders['jenis_pinjaman'].classes_)
            tujuan_penggunaan = st.selectbox('Jenis Penggunaan', encoders['tujuan_penggunaan'].classes_)
            plafon = st.number_input('Plafon (juta Rupiah)', 0., None, value=271000.)
            tenor = st.number_input('Tenor (bulan)', 3, None, value=60)
            nilai_agunan = st.number_input('Nilai Pasar Agunan (juta Rupiah)', 0., None, value=300.)
            sandi_pengikatan = st.selectbox('Sandi Pengikatan', encoders['sandi_pengikatan'].classes_)
            restruktur = st.radio('Restruktur', options=['False', 'True'], horizontal=True)

        submitted = st.form_submit_button('Submit', use_container_width=True)

    if submitted:
        X = [
            nilai_agunan, dbr, plafon, nilai_agunan/plafon, tenor, usia,
            encoders['jenis_pinjaman'].transform([jenis_pinjaman]).item(),
            encoders['alamat_kota'].transform([alamat_kota]).item(),
            encoders['Penilaian Rekening Koran'].transform([rek_koran]).item(),
            encoders['tujuan_penggunaan'].transform([tujuan_penggunaan]).item(),
            encoders['gaji(Penghasilan)'][gaji],
            encoders['pendidikan'][pendidikan],
            encoders['sandi_pengikatan'].transform([sandi_pengikatan]).item(),
            encoders['Riwayat Slik'].transform([riwayat_slik]).item(),
            encoders['jenis_usaha'].transform([jenis_usaha]).item(),
            encoders['Restruktur'].transform([restruktur]).item()
        ]
        pred = model.predict([X])[0]
        probs = model.predict_proba([X])[0]
        risk_lvl = ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Cukup Tinggi', 'Sangat Tinggi']
        color_lvl = ['#44ce1b', '#deff8b', '#f7e379', '#f2a134', '#ff4545', '#e51f1f']

        st.markdown(f"""
            <h4 style="text-align: center;">
                Berdasarkan input di atas, debitur diprediksi memiliki risiko
                <strong style="color: {color_lvl[int(pred-1)]};">{risk_lvl[int(pred)-1]}</strong>
            </h4>
        """, unsafe_allow_html=True)

        st.divider()

        sv = explainer(np.array(X).reshape(1, -1))
        shap_values = shap.Explanation(
            sv.values[:, :, int(pred)-1],
            sv.base_values[:, int(pred)-1],
            data=np.array(X).reshape(1, -1),
            feature_names=model.feature_name_
        )

        shap.plots.waterfall(shap_values[0], max_display=len(X), show=False)
        fig = plt.gcf()
        fig.set_size_inches(20, 12)
        plt.tight_layout()
        st.pyplot(fig)


# BODY 3 — Risk Scoring (2): Tree Viz
def body_3():
    st.markdown("<h3>Visualisasi Struktur Pohon Keputusan</h3>", unsafe_allow_html=True)

    st.write("Berikut visualisasi salah satu pohon dari model LightGBM (jika tersedia):")

    try:
        # LightGBM tidak langsung menyediakan estimator_ seperti RandomForest
        # Maka kita ambil visualisasi dari booster
        import lightgbm as lgb
        ax = lgb.plot_tree(model, tree_index=0, figsize=(40, 20), show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
        st.pyplot(ax.figure)
    except Exception as e:
        st.error(f"Visualisasi pohon gagal ditampilkan: {e}")


# NAVBAR
def navbar_menu():
    with st.sidebar:
        selected_menu = option_menu(
            menu_title='Menu',
            options=['Risk Scoring (1)', 'Risk Scoring (2)'],
            icons=['1-square', '2-square'],
            menu_icon='cast',
            default_index=0,
            orientation='vertical'
        )
        st.markdown('<hr>', unsafe_allow_html=True)
    return selected_menu


# MAIN
def main():
    selected_menu = navbar_menu()
    if selected_menu == 'Risk Scoring (1)':
        body_2()
    elif selected_menu == 'Risk Scoring (2)':
        body_3()


# RUN
if __name__ == '__main__':
    set_page_config()
    main()
