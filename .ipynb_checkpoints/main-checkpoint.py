import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import shap


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


# BODY 1
def body_1():
    pass


# BODY 2
def body_2():
    with st.form('Input Data Debitur'):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h4 style='text-align: center;'>Profil Debitur</h4>", unsafe_allow_html=True)
            st.divider()

            usia = st.number_input(
                label='Usia (tahun)',
                min_value=18, max_value=80,
                value=21
            )
            alamat_kota = st.selectbox(
                label='Alamat Kota',
                options=encoders['alamat_kota'].classes_
            )
            pendidikan = st.selectbox(
                label='Pendidikan',
                options=encoders['pendidikan'].keys()
            )
            jenis_usaha = st.selectbox(
                label='Jenis Usaha',
                options=encoders['jenis_usaha'].classes_
            )
            gaji = st.select_slider(
                label='Penghasilan',
                options=encoders['gaji(Penghasilan)'].keys(),
                value='> 7,500,000 S/D 10,000,000'
            )

        with col2:
            st.markdown("<h4 style='text-align: center;'>Riwayat Debitur</h4>", unsafe_allow_html=True)
            st.divider()

            riwayat_slik = st.selectbox(
                label='Riwayat SLIK',
                options=encoders['Riwayat Slik'].classes_
            )
            rek_koran = st.selectbox(
                label='Penilaian Rek. Koran',
                options=encoders['Penilaian Rekening Koran'].classes_
            )
            dbr = st.number_input(
                label='Debt Burden Ratio (%)',
                min_value=0., max_value=100.,
                value=50.0, step=0.1
            )

        with col3:
            st.markdown("<h4 style='text-align: center;'>Ket. Pinjaman</h4>", unsafe_allow_html=True)
            st.divider()

            jenis_pinjaman = st.selectbox(
                label='Jenis Pinjaman',
                options=encoders['jenis_pinjaman'].classes_
            )
            tujuan_penggunaan = st.selectbox(
                label='Jenis Penggunaan',
                options=encoders['tujuan_penggunaan'].classes_
            )
            plafon = st.number_input(
                label='Plafon (juta Rupiah)',
                min_value=0., max_value=None,
                value=271000., step=1.
            )
            tenor = st.number_input(
                label='Tenor (bulan)',
                min_value=3, max_value=None,
                value=60, step=1
            )
            nilai_agunan = st.number_input(
                label='Nilai Pasar Agunan (juta Rupiah)',
                min_value=0., max_value=None,
                value=300., step=1.
            )
            sandi_pengikatan = st.selectbox(
                label='Sandi Pengikatan',
                options=encoders['sandi_pengikatan'].classes_
            )
            restruktur = st.segmented_control(
                label='Restruktur',
                options=['False', 'True'],
                default='False'
            )

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
        df = pd.DataFrame({
            'risk': risk_lvl,
            'probs': probs,
        })
        df['condition'] = df['probs']==df['probs'].max()

        # base = alt.Chart(df).encode(
        #     y=alt.Y('risk:N', title='', axis=alt.Axis(labelAngle=0), sort=risk_lvl),
        #     x=alt.X('probs:Q', title='Probabilitas', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(grid=False, domain=True, tickCount=10, format='.0%')),
        #     text=alt.Text('probs:Q', format=".2%")
        # )
        # bars = base.mark_bar().encode(
        #     color=alt.Color(
        #         'risk:N',
        #         scale=alt.Scale(domain=risk_lvl, range=['#44ce1b', '#deff8b', '#f7e379', '#f2a134', '#ff4545', '#e51f1f']),
        #         legend=None
        #     )
        # )
        # text = base.mark_text(align='left', dx=10).encode(color=alt.value('white'))
        # chart = (bars + text).properties(title='Prediksi Tingkat Resiko Debitur', width=500, height=400)
        #
        color_lvl = ['#44ce1b', '#deff8b', '#f7e379', '#f2a134', '#ff4545', '#e51f1f']

        st.write('\n')
        st.markdown(f'''
            <h4 style="text-align: center;">
                Berdasarkan input di atas, debitur bersangkutan diprediksi memiliki resiko
                <strong style="color: {color_lvl[int(pred-1)]};">{risk_lvl[int(pred) - 1]}</strong> dalam peminjaman.
            </h4>
        ''', unsafe_allow_html=True)

        st.write('\n')
        st.divider()
        # st.altair_chart(chart, use_container_width=True)

        sv = explainer(np.array(X).reshape(1, -1))
        shap_values = shap.Explanation(sv.values[:, :, int(pred)-1],
                               sv.base_values[:, int(pred)-1],
                               data=np.array(X).reshape(1, -1),
                               feature_names=model.feature_name_)

        shap.plots.waterfall(shap_values[0], max_display=len(X), show=False)
        fig = plt.gcf()
        fig.set_size_inches(20, 12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)




# BODY 3
def body_3():
    pass


# NAVBAR
def navbar_menu():
    with st.sidebar:
        selected_menu = option_menu(
            menu_title='Menu',
            options=['Data Insight', 'Risk Scoring (1)', 'Risk Scoring (2)'],
            icons=['1-square', '2-square', '3-square'],
            menu_icon='cast',
            default_index=1,
            orientation='vertical'
        )

        st.markdown('<hr>', unsafe_allow_html=True)

    return selected_menu


# MAIN
def main():
    selected_menu = navbar_menu()
    if selected_menu == 'Data Insight':
        st.title('To Be Arranged')
        body_1()
    elif selected_menu == 'Risk Scoring (2)':
        st.title('To Be Arranged')
        body_3()
    else:
        st.markdown("<h1 style='text-align: center;'>Model Risk Scoring 1</h1>", unsafe_allow_html=True)
        body_2()


# RUN PROGRAM
if __name__ == '__main__':
    set_page_config()
    main()