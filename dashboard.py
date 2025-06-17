import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns # Masih digunakan untuk sns.color_palette
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# --- Layout Dashboard Streamlit ---
# set_page_config() must be called as the first Streamlit command in your script.
st.set_page_config(layout="wide") # Menggunakan layout lebar untuk tampilan yang lebih baik

# --- Load Data ---
# Pastikan 'dataset_all.csv' ada di direktori yang sama dengan script Streamlit Anda.
@st.cache_data
def load_data():
    df = pd.read_csv('dataset_all.csv')
    df['Jumlah'] = pd.to_numeric(df['Jumlah'], errors='coerce')
    df['Tahun'] = pd.to_numeric(df['Tahun'], errors='coerce')
    df = df.dropna(subset=['Jumlah', 'Tahun'])
    df['Tahun'] = df['Tahun'].astype(int)
    return df

df = load_data()

st.title("Dashboard Analisis Data Anak Putus Sekolah Kota Bandar Lampung")
st.write("Analisis ini bertujuan untuk memahami tren anak putus sekolah berdasarkan jenjang, jenis kelamin, dan perubahan tahunan di Kota Bandar Lampung.")

# Sidebar untuk filter dan navigasi
st.sidebar.header("Filter & Navigasi")
selected_analysis = st.sidebar.radio(
    "Pilih Tipe Analisis:",
    options=[
        'Tren Tahunan per Jenjang & Jenis Kelamin',
        'Jenjang dengan Angka Putus Sekolah Tertinggi',
        'Perbedaan Putus Sekolah Berdasarkan Jenis Kelamin',
        'Distribusi Jenjang dan Jenis Kelamin',
        'Analisis Lonjakan/Penurunan Tahunan'
    ],
    index=0 # Default selected option
)

# Konten utama dashboard
st.markdown("---") # Garis pemisah untuk estetika

if df.empty:
    st.warning("Tidak ada data untuk ditampilkan. Pastikan 'dataset_all.csv' ada dan valid.")
else:
    # --- 1. Tren Tahunan per Jenjang & Jenis Kelamin ---
    if selected_analysis == 'Tren Tahunan per Jenjang & Jenis Kelamin':
        st.header("1. Tren Jumlah Anak Putus Sekolah per Jenjang dan Jenis Kelamin")
        total_pertahun_gender = df.groupby(['Tahun', 'Jenjang', 'Jenis Kelamin'])['Jumlah'].sum().reset_index()

        if not total_pertahun_gender.empty:
            fig = px.line(
                total_pertahun_gender,
                x='Tahun',
                y='Jumlah',
                color='Jenjang',
                line_dash='Jenis Kelamin',
                markers=True,
                title="Tren Jumlah Anak Putus Sekolah per Jenjang dan Jenis Kelamin di Bandar Lampung",
                labels={"Jumlah": "Jumlah Anak Putus Sekolah"}
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Model Regresi Linear per Jenjang (Total Gender)")

            jenjang_options = df['Jenjang'].unique().tolist()
            selected_jenjang_for_regresi = st.selectbox(
                "Pilih Jenjang untuk Regresi Linear:",
                options=jenjang_options,
                index=0 if len(jenjang_options) > 0 else None
            )

            if selected_jenjang_for_regresi:
                total_pertahun_gender_agg = df.groupby(['Tahun', 'Jenjang', 'Jenis Kelamin'])['Jumlah'].sum().reset_index()
                df_jenjang = total_pertahun_gender_agg[total_pertahun_gender_agg['Jenjang'] == selected_jenjang_for_regresi]

                X = df_jenjang[['Tahun']].drop_duplicates()
                # Pastikan y adalah total jumlah per tahun untuk jenjang yang dipilih, tanpa membedakan jenis kelamin
                y = df_jenjang.groupby('Tahun')['Jumlah'].sum().reset_index()['Jumlah']

                if not X.empty and not y.empty and len(X) > 1: # Minimal 2 titik data untuk regresi
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)

                    mae = mean_absolute_error(y, y_pred)
                    r2 = r2_score(y, y_pred)

                    fig_reg = go.Figure()
                    fig_reg.add_trace(go.Scatter(x=X['Tahun'], y=y, mode='lines+markers', name='Aktual'))
                    fig_reg.add_trace(go.Scatter(x=X['Tahun'], y=y_pred, mode='lines', name='Prediksi', line=dict(color='red', dash='dash')))
                    fig_reg.update_layout(
                        title=f'Regresi Linear - {selected_jenjang_for_regresi} (Total Gender) | MAE: {mae:.2f}, R²: {r2:.2f}',
                        xaxis_title='Tahun',
                        yaxis_title='Jumlah Anak Putus Sekolah',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_reg, use_container_width=True)
                else:
                    st.info(f"Tidak ada data yang cukup (minimal 2 tahun) untuk regresi jenjang '{selected_jenjang_for_regresi}'.")
            else:
                st.info("Pilih jenjang dari dropdown untuk melihat regresi linear.")
        else:
            st.info("Tidak ada data untuk menampilkan Tren Tahunan per Jenjang dan Jenis Kelamin.")

    # --- 2. Jenjang dengan Angka Putus Sekolah Tertinggi ---
    elif selected_analysis == 'Jenjang dengan Angka Putus Sekolah Tertinggi':
        st.header("2. Jenjang Pendidikan dengan Jumlah Anak Putus Sekolah Paling Tinggi Tiap Tahunnya")
        total_pertahun = df.groupby(['Tahun', 'Jenjang'])['Jumlah'].sum().reset_index()
        if not total_pertahun.empty:
            max_jenjang_per_year = total_pertahun.loc[total_pertahun.groupby("Tahun")["Jumlah"].idxmax()]
            st.dataframe(max_jenjang_per_year, use_container_width=True)
        else:
            st.info("Tidak ada data untuk menampilkan Jenjang dengan Angka Putus Sekolah Tertinggi.")

    # --- 3. Perbedaan Putus Sekolah Berdasarkan Jenis Kelamin ---
    elif selected_analysis == 'Perbedaan Putus Sekolah Berdasarkan Jenis Kelamin':
        st.header("3. Perbedaan Jumlah Anak Putus Sekolah Berdasarkan Jenis Kelamin")
        gender_yearly = df.groupby(['Tahun', 'Jenis Kelamin'])['Jumlah'].sum().reset_index()

        if not gender_yearly.empty:
            fig = px.line(
                gender_yearly,
                x='Tahun',
                y='Jumlah',
                color='Jenis Kelamin',
                markers=True,
                title="Perbedaan Jumlah Anak Putus Sekolah Berdasarkan Jenis Kelamin",
                labels={"Jumlah": "Jumlah"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak ada data untuk menampilkan Perbedaan Putus Sekolah Berdasarkan Jenis Kelamin.")

    # --- 4. Pola Distribusi Jenjang dan Jenis Kelamin ---
    elif selected_analysis == 'Distribusi Jenjang dan Jenis Kelamin':
        st.header("4. Pola Distribusi Anak Putus Sekolah Berdasarkan Kombinasi Jenjang & Jenis Kelamin")
        if not df.empty:
            # Generate a palette using seaborn for consistent colors (not directly used by px, but good for understanding)
            # jenjang_gender_palette = sns.color_palette("viridis", n_colors=len(df['Jenis Kelamin'].unique()) * len(df['Jenjang'].unique())).as_hex()

            fig = px.bar(
                df.groupby(['Jenjang', 'Jenis Kelamin'])['Jumlah'].sum().reset_index(),
                x='Jenjang',
                y='Jumlah',
                color='Jenis Kelamin',
                barmode='group',
                title="Distribusi Anak Putus Sekolah Berdasarkan Jenjang dan Jenis Kelamin",
                labels={"Jumlah": "Total Jumlah"}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Pie chart untuk distribusi berdasarkan Jenjang
            st.subheader("Distribusi Total Anak Putus Sekolah Berdasarkan Jenjang")
            total_per_jenjang = df.groupby('Jenjang')['Jumlah'].sum().reset_index()
            fig_pie_jenjang = px.pie(
                total_per_jenjang,
                values='Jumlah',
                names='Jenjang',
                title='Proporsi Anak Putus Sekolah per Jenjang',
                hole=0.3 # Membuat donut chart
            )
            st.plotly_chart(fig_pie_jenjang, use_container_width=True)

            st.subheader("Tabel Detail Distribusi")
            jumlah_per_jenjang_gender = df.groupby(['Jenjang', 'Jenis Kelamin'])['Jumlah'].sum().reset_index()
            st.dataframe(jumlah_per_jenjang_gender, use_container_width=True)
        else:
            st.info("Tidak ada data untuk menampilkan Distribusi Jenjang dan Jenis Kelamin.")

    # --- 5. Analisis Lonjakan/Penurunan Tahunan ---
    elif selected_analysis == 'Analisis Lonjakan/Penurunan Tahunan':
        st.header("5. Tahun dengan Lonjakan atau Penurunan Tajam Jumlah Anak Putus Sekolah")
        total_semua = df.groupby('Tahun')['Jumlah'].sum().reset_index()

        if not total_semua.empty:
            df_total_per_year = total_semua.copy()
            df_total_per_year['Delta'] = df_total_per_year['Jumlah'].diff()
            df_total_per_year['Delta'] = df_total_per_year['Delta'].fillna(0)
            df_total_per_year['Tahun'] = df_total_per_year['Tahun'].astype(int)

            # Hanya Line chart untuk perubahan jumlah
            st.subheader("Tren Perubahan Jumlah Anak Putus Sekolah per Tahun")
            fig_line_delta = px.line(
                df_total_per_year,
                x='Tahun',
                y='Delta',
                markers=True,
                title="Tren Perubahan Jumlah Anak Putus Sekolah per Tahun",
                labels={"Delta": "Perubahan Jumlah (Delta)"},
                line_shape='linear'
            )
            fig_line_delta.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig_line_delta, use_container_width=True)


            st.subheader("Detail Perubahan Tahunan")
            st.dataframe(df_total_per_year[['Tahun', 'Jumlah', 'Delta']], use_container_width=True)

            # Menemukan lonjakan dan penurunan terbesar, pastikan data tidak kosong
            if not df_total_per_year.empty:
                max_increase = df_total_per_year[df_total_per_year['Delta'] == df_total_per_year['Delta'].max()]
                max_decrease = df_total_per_year[df_total_per_year['Delta'] == df_total_per_year['Delta'].min()]

                st.subheader("Lonjakan Terbesar")
                st.dataframe(max_increase, use_container_width=True)

                st.subheader("Penurunan Terbesar")
                st.dataframe(max_decrease, use_container_width=True)

            else:
                st.info("Tidak ada perubahan tahunan yang dapat dianalisis.")
        else:
            st.info("Tidak ada data untuk menampilkan Analisis Lonjakan/Penurunan Tahunan.")