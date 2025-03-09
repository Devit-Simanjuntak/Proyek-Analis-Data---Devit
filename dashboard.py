import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    # Memuat data dari file CSV
    tiantan_df = pd.read_csv("data/PRSA_Data_Tiantan_20130301-20170228.csv")
    
    # Membersihkan data - menangani missing values
    tiantan_df.fillna(method='ffill', inplace=True)
    
    # Membuat kolom datetime untuk analisis time series
    tiantan_df['datetime'] = pd.to_datetime(tiantan_df[['year', 'month', 'day', 'hour']])
    
    # Mendefinisikan musim berdasarkan bulan
    tiantan_df['season'] = pd.cut(
        tiantan_df['month'], 
        bins=[0, 3, 6, 9, 12], 
        labels=['Winter', 'Spring', 'Summer', 'Fall'],
        include_lowest=True
    )
    
    # Menambahkan kolom hari dalam seminggu
    tiantan_df['day_of_week'] = tiantan_df['datetime'].dt.dayofweek
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    tiantan_df['day_name'] = tiantan_df['day_of_week'].map(dict(zip(range(7), day_names)))
    
    return tiantan_df

# Fungsi untuk menghitung AQI sederhana
def calculate_simple_aqi(row, who_limits):
    pm25_aqi = row['PM2.5'] / who_limits['PM2.5']
    pm10_aqi = row['PM10'] / who_limits['PM10']
    no2_aqi = row['NO2'] / who_limits['NO2']
    # Mengembalikan nilai maksimum yang sudah dinormalisasi
    return max(pm25_aqi, pm10_aqi, no2_aqi)

# Fungsi untuk mengklasifikasikan AQI
def classify_aqi(aqi):
    if aqi <= 1:
        return 'Good'
    elif aqi <= 2:
        return 'Moderate'
    elif aqi <= 3:
        return 'Unhealthy for Sensitive Groups'
    elif aqi <= 4:
        return 'Unhealthy'
    else:
        return 'Very Unhealthy'

# Main function
def main():
    # Set page config
    st.set_page_config(page_title="Air Quality Dashboard", page_icon="🌬️", layout="wide")
    
    # Judul dashboard
    st.title("🌬️ Dashboard Analisis Kualitas Udara di Tiantan (2013-2017)")
    st.subheader("Oleh: Devit Imanuel Nuary Simanjuntak")
    
    # Sidebar
    st.sidebar.title("Navigasi")
    pages = ["Pendahuluan", 
             "Q1: Variasi Polusi per Musim", 
             "Q2: Hubungan Cuaca & Kualitas Udara",
             "Q3: Pola Polusi Harian & Jam",
             "Q4: Tren Kualitas Udara 2013-2017"]
    selected_page = st.sidebar.radio("Pilih Halaman", pages)
    
    # Memuat dan menyimpan data
    try:
        data = load_data()
        
        # Definisi variabel yang akan sering digunakan
        pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
        weather_vars = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        
        # WHO guidelines
        who_limits = {
            'PM2.5': 25,
            'PM10': 50,
            'NO2': 40,
            'SO2': 20,
            'O3': 100
        }
        
        # Page content
        if selected_page == "Pendahuluan":
            show_introduction(data, pollutants, weather_vars)
        elif selected_page == "Q1: Variasi Polusi per Musim":
            show_seasonal_variation(data, pollutants)
        elif selected_page == "Q2: Hubungan Cuaca & Kualitas Udara":
            show_weather_relationship(data, pollutants, weather_vars)
        elif selected_page == "Q3: Pola Polusi Harian & Jam":
            show_time_patterns(data, pollutants)
        elif selected_page == "Q4: Tren Kualitas Udara 2013-2017":
            show_yearly_trends(data, pollutants, who_limits)
            
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam memuat data: {e}")
        st.info("Pastikan file CSV 'PRSA_Data_Tiantan_20130301-20170228.csv' tersedia di direktori yang sama.")

# Fungsi untuk halaman pendahuluan
def show_introduction(data, pollutants, weather_vars):
    st.header("Pendahuluan")
    
    # Pertanyaan bisnis
    st.subheader("Pertanyaan Bisnis")
    st.markdown("""
    - Bagaimana polusi udara bervariasi setiap musim, dan musim mana yang memiliki kualitas udara terburuk?
    - Apa hubungan antara kondisi cuaca dengan kualitas udara?
    - Apa pola polusi udara yang terjadi setiap jam dan setiap hari, dan bagaimana informasi ini membantu untuk melakukan perencanaan aktivitas luar ruangan?
    - Apakah kualitas udara membaik atau memburuk dalam kurun waktu 5 tahun (2013-2017)?
    """)
    
    # Tampilkan sampel data
    st.subheader("Sampel Data")
    st.dataframe(data.head())
    
    # Tampilkan informasi statistik
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Statistik Deskriptif")
        st.dataframe(data[pollutants].describe())
    
    with col2:
        st.subheader("Informasi Dataset")
        st.write(f"Jumlah Baris: {data.shape[0]}")
        st.write(f"Jumlah Kolom: {data.shape[1]}")
        st.write(f"Periode Data: {data['datetime'].min().date()} hingga {data['datetime'].max().date()}")
    
    # Tampilkan matriks korelasi
    st.subheader("Analisis Korelasi")
    correlation_variables = pollutants + weather_vars
    correlation_matrix = data[correlation_variables].corr(method="pearson").round(2)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    plt.title('Korelasi Antara Polutan dan Variabel Cuaca')
    st.pyplot(fig)

# Fungsi untuk halaman variasi musiman
def show_seasonal_variation(data, pollutants):
    st.header("Variasi Polusi Udara per Musim")
    st.markdown("""
    Analisis ini menunjukkan bagaimana tingkat polutan berbeda di setiap musim
    dan mengidentifikasi musim dengan kualitas udara terburuk.
    """)
    
    # Menghitung rata-rata harian untuk tiap polutan berdasarkan musim
    daily_avg = data.groupby(['year', 'month', 'day', 'season'])[pollutants].mean().reset_index()
    
    # Plot boxplot untuk setiap polutan berdasarkan musim
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, pollutant in enumerate(pollutants):
        sns.boxplot(x='season', y=pollutant, data=daily_avg, ax=axes[i])
        axes[i].set_title(f'Distribusi {pollutant} per Musim')
        axes[i].set_xlabel('Musim')
        axes[i].set_ylabel('Konsentrasi')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Menghitung rata-rata polutan berdasarkan musim
    season_avg = daily_avg.groupby('season')[pollutants].mean().reset_index()
    
    st.subheader("Rata-rata Tingkat Polutan per Musim")
    st.dataframe(season_avg.round(2))
    
    # Visualisasi dengan bar chart
    st.subheader("Perbandingan Tingkat Polutan per Musim")
    
    selected_pollutant = st.selectbox("Pilih Polutan untuk Visualisasi", pollutants)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='season', y=selected_pollutant, data=season_avg, ax=ax)
    ax.set_title(f'Rata-rata {selected_pollutant} per Musim')
    ax.set_xlabel('Musim')
    ax.set_ylabel('Konsentrasi')
    st.pyplot(fig)
    
    # Kesimpulan
    st.subheader("Kesimpulan")
    st.markdown("""
    - Polusi udara pada umumnya cenderung tinggi pada musim dingin (Winter) dan musim gugur (Fall), kecuali untuk O3.
    - O3 (Ozon) memiliki pola berbeda dengan polutan lain, dengan konsentrasi tertinggi pada musim panas (Summer).
    - Musim yang memiliki kualitas udara terburuk secara keseluruhan adalah musim gugur diikuti musim dingin.
    - Musim semi (Spring) secara konsisten memiliki tingkat polusi yang lebih rendah untuk hampir semua polutan.
    """)

# Fungsi untuk halaman hubungan cuaca dan kualitas udara
def show_weather_relationship(data, pollutants, weather_vars):
    st.header("Hubungan antara Kondisi Cuaca dengan Kualitas Udara")
    
    # Hitung korelasi
    correlation_variables = pollutants + weather_vars
    correlation_matrix = data[correlation_variables].corr(method="pearson").round(2)
    
    # Tampilkan heatmap korelasi
    st.subheader("Matriks Korelasi: Polutan vs Variabel Cuaca")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    plt.title('Korelasi Antara Polutan dan Variabel Cuaca')
    st.pyplot(fig)
    
    # Scatter plots untuk variabel cuaca dan polutan pilihan
    st.subheader("Hubungan Detail antara Cuaca dan Polutan")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_weather = st.selectbox("Pilih Variabel Cuaca", weather_vars)
    with col2:
        selected_pollutant = st.selectbox("Pilih Polutan", pollutants, key="pollutant_selector_2")
    
    # Membuat scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=selected_weather, y=selected_pollutant, data=data, alpha=0.3, ax=ax)
    sns.regplot(x=selected_weather, y=selected_pollutant, data=data, scatter=False, color='red', ax=ax)
    ax.set_title(f'{selected_weather} vs {selected_pollutant}')
    ax.set_xlabel(selected_weather)
    ax.set_ylabel(selected_pollutant)
    st.pyplot(fig)
    
    # Analisis arah angin
    st.subheader("Dampak Arah Angin pada Polutan")
    
    wind_pollution = data.groupby('wd')[pollutants].mean().reset_index()
    selected_pollutant_wind = st.selectbox("Pilih Polutan untuk Analisis Arah Angin", pollutants, key="wind_pollutant")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='wd', y=selected_pollutant_wind, data=wind_pollution, ax=ax)
    ax.set_title(f'Rata-rata {selected_pollutant_wind} berdasarkan Arah Angin')
    ax.set_xlabel('Arah Angin')
    ax.set_ylabel('Konsentrasi')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    
    # Kesimpulan
    st.subheader("Kesimpulan")
    st.markdown("""
    - Suhu (TEMP) berkorelasi negatif dengan beberapa polutan seperti PM2.5, PM10, dan SO2, artinya polusi cenderung lebih tinggi saat suhu rendah.
    - Kecepatan angin (WSPM) berkorelasi negatif dengan hampir semua polutan kecuali O3, menunjukkan angin kencang dapat membantu menyebarkan polutan.
    - Curah hujan (RAIN) memiliki korelasi negatif lemah dengan polutan partikulat, mengindikasikan hujan dapat membantu 'mencuci' polutan dari udara.
    - Arah angin tertentu dapat membawa lebih banyak polutan, tergantung pada lokasi sumber polusi di sekitar stasiun pemantauan.
    - Secara umum, kondisi cuaca dengan kualitas udara cenderung memiliki hubungan yang negatif.
    """)

# Fungsi untuk halaman pola waktu
def show_time_patterns(data, pollutants):
    st.header("Pola Polusi Udara berdasarkan Waktu")
    st.markdown("""
    Analisis ini menunjukkan bagaimana tingkat polutan bervariasi sepanjang hari dan minggu,
    serta memberikan rekomendasi untuk waktu terbaik melakukan aktivitas luar ruangan.
    """)
    
    # Pola setiap jam
    st.subheader("Pola Polusi Udara per Jam")
    hourly_avg = data.groupby('hour')[pollutants].mean()
    
    selected_pollutant_hourly = st.selectbox("Pilih Polutan untuk Pola per Jam", pollutants)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    hourly_avg[selected_pollutant_hourly].plot(ax=ax)
    ax.set_title(f'Pola {selected_pollutant_hourly} per Jam')
    ax.set_xlabel('Jam (0-23)')
    ax.set_ylabel('Konsentrasi')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True)
    st.pyplot(fig)
    
    # Untuk melihat semua polutan sekaligus
    if st.checkbox("Tampilkan pola semua polutan per jam"):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, pollutant in enumerate(pollutants):
            hourly_avg[pollutant].plot(ax=axes[i])
            axes[i].set_title(f'Pola {pollutant} per Jam')
            axes[i].set_xlabel('Jam (0-23)')
            axes[i].set_ylabel('Konsentrasi')
            axes[i].set_xticks(range(0, 24, 3))
            axes[i].grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Pola setiap hari
    st.subheader("Pola Polusi Udara per Hari dalam Seminggu")
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_patterns = data.groupby('day_name')[pollutants].mean()
    daily_patterns = daily_patterns.reindex(day_names)
    
    selected_pollutant_daily = st.selectbox("Pilih Polutan untuk Pola Harian", pollutants, key="daily_pollutant")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=daily_patterns.index, y=daily_patterns[selected_pollutant_daily], ax=ax)
    ax.set_title(f'Pola {selected_pollutant_daily} per Hari dalam Seminggu')
    ax.set_xlabel('Hari')
    ax.set_ylabel('Konsentrasi')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    
    # Rekomendasi waktu untuk aktivitas luar ruangan
    st.subheader("Rekomendasi Waktu untuk Aktivitas Luar Ruangan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Jam Terbaik (Polusi Terendah):**")
        best_hours = hourly_avg[['PM2.5', 'PM10', 'O3']].idxmin()
        for pollutant, hour in best_hours.items():
            st.write(f"- {pollutant}: {hour}:00")
            
        st.markdown("**Hari Terbaik (Polusi Terendah):**")
        best_days = daily_patterns[['PM2.5', 'PM10', 'O3']].idxmin()
        for pollutant, day in best_days.items():
            st.write(f"- {pollutant}: {day}")
    
    with col2:
        st.markdown("**Jam Terburuk (Polusi Tertinggi):**")
        worst_hours = hourly_avg[['PM2.5', 'PM10', 'O3']].idxmax()
        for pollutant, hour in worst_hours.items():
            st.write(f"- {pollutant}: {hour}:00")
    
    # Kesimpulan
    st.subheader("Kesimpulan")
    st.markdown("""
    - Polusi udara cenderung tinggi di jam malam dan dini hari (jam 16.00 sampai dengan jam 02.00) dan agak meningkat pada jam 06.00 - 10.00.
    - Jam terbaik untuk aktivitas luar ruangan umumnya adalah di siang hari (12.00-15.00) ketika level PM2.5 dan PM10 terendah.
    - Untuk O3 (Ozon), pola berbeda dengan konsentrasi terendah di pagi hari dan tertinggi di siang hari.
    - Tingkat polusi udara cenderung tinggi pada akhir pekan yakni hari Sabtu dan Minggu.
    - Hari yang terbaik untuk melakukan kegiatan di luar ruangan umumnya adalah hari Senin.
    """)

# Fungsi untuk halaman tren tahunan
def show_yearly_trends(data, pollutants, who_limits):
    st.header("Tren Kualitas Udara (2013-2017)")
    
    # Menghitung rata-rata tahunan
    yearly_avg = data.groupby('year')[pollutants].mean()
    yearly_std = data.groupby('year')[pollutants].std()
    
    # Plot tren tahunan
    st.subheader("Tren Polutan per Tahun")
    
    selected_pollutant_yearly = st.selectbox("Pilih Polutan untuk Tren Tahunan", pollutants)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    yearly_avg[selected_pollutant_yearly].plot(marker='o', linestyle='-', linewidth=2, ax=ax)
    
    # Add error bars (standard deviation)
    ax.fill_between(
        yearly_avg.index,
        yearly_avg[selected_pollutant_yearly] - yearly_std[selected_pollutant_yearly],
        yearly_avg[selected_pollutant_yearly] + yearly_std[selected_pollutant_yearly],
        alpha=0.3
    )
    
    ax.set_title(f'Tren Tahunan {selected_pollutant_yearly} (2013-2017)')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Konsentrasi')
    ax.grid(True)
    st.pyplot(fig)
    
    # Untuk melihat semua tren sekaligus
    if st.checkbox("Tampilkan tren semua polutan"):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, pollutant in enumerate(pollutants):
            yearly_avg[pollutant].plot(marker='o', linestyle='-', linewidth=2, ax=axes[i])
            
            # Add error bars
            axes[i].fill_between(
                yearly_avg.index,
                yearly_avg[pollutant] - yearly_std[pollutant],
                yearly_avg[pollutant] + yearly_std[pollutant],
                alpha=0.3
            )
            
            axes[i].set_title(f'Tren {pollutant} (2013-2017)')
            axes[i].set_xlabel('Tahun')
            axes[i].set_ylabel('Konsentrasi')
            axes[i].grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Persentase perubahan
    pct_change = ((yearly_avg.loc[2017] - yearly_avg.loc[2013]) / yearly_avg.loc[2013] * 100).round(2)
    
    st.subheader("Persentase Perubahan Polutan (2013-2017)")
    
    # Visualisasi persentase perubahan dengan bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    pct_change.plot(kind='bar', ax=ax)
    ax.set_title('Persentase Perubahan Tingkat Polutan (2013-2017)')
    ax.set_xlabel('Polutan')
    ax.set_ylabel('Perubahan (%)')
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Memberikan warna berbeda untuk nilai positif dan negatif
    for i, v in enumerate(pct_change):
        color = 'green' if v < 0 else 'red'
        ax.get_children()[i].set_color(color)
    
    st.pyplot(fig)
    
    # Air Quality Index (AQI)
    st.subheader("Analisis Air Quality Index (AQI)")
    
    # Menghitung rata-rata harian untuk analisis AQI
    daily_data = data.groupby(['year', 'month', 'day'])[pollutants].mean().reset_index()
    
    # Menghitung AQI
    daily_data['AQI'] = daily_data.apply(lambda row: calculate_simple_aqi(row, who_limits), axis=1)
    daily_data['AQI_Category'] = daily_data['AQI'].apply(classify_aqi)
    
    # Rata-rata AQI tahunan
    yearly_aqi = daily_data.groupby('year')['AQI'].mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    yearly_aqi.plot(kind='bar', color='teal', ax=ax)
    ax.set_title('Rata-rata Air Quality Index per Tahun')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Nilai AQI (Semakin Tinggi Semakin Buruk)')
    ax.grid(axis='y')
    st.pyplot(fig)
    
    # Distribusi kategori AQI
    aqi_categories = daily_data.groupby(['year', 'AQI_Category']).size().unstack().fillna(0)
    aqi_categories_pct = aqi_categories.div(aqi_categories.sum(axis=1), axis=0) * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    aqi_categories_pct.plot(kind='bar', stacked=True, colormap='viridis', ax=ax)
    ax.set_title('Distribusi Kategori Kualitas Udara per Tahun')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Persentase (%)')
    ax.legend(title='Kategori AQI')
    ax.grid(axis='y')
    st.pyplot(fig)
    
    # Persentase hari yang melewati batas WHO
    st.subheader("Persentase Hari Melebihi Batas WHO")
    
    # Menghitung hari yang melewati batas WHO
    exceeding_days = {}
    for pollutant, limit in who_limits.items():
        yearly_counts = daily_data[daily_data[pollutant] > limit].groupby('year').size()
        yearly_total = daily_data.groupby('year').size()
        yearly_percentage = (yearly_counts / yearly_total * 100).round(1)
        exceeding_days[pollutant] = yearly_percentage
    
    exceeding_df = pd.DataFrame(exceeding_days)
    
    # Visualisasi dengan bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    exceeding_df.plot(kind='bar', ax=ax)
    ax.set_title('Persentase Hari Melebihi Batas WHO (2013-2017)')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Persentase Hari (%)')
    ax.legend(title='Polutan')
    ax.grid(axis='y')
    st.pyplot(fig)
    
    # Kesimpulan
    st.subheader("Kesimpulan")
    st.markdown("""
    - Kualitas udara sempat membaik pada tahun 2013 sampai 2016 untuk sebagian besar polutan.   
    - Namun, tingkat polusi meningkat kembali dari tahun 2016 ke tahun 2017, terutama untuk PM2.5 dan PM10.
    - Persentase hari yang melebihi batas WHO masih tinggi, khususnya untuk PM2.5 dan PM10.
    - Berdasarkan AQI, terjadi peningkatan hari-hari dengan kualitas udara yang "Unhealthy" di tahun 2017.
    - Meskipun ada perbaikan untuk beberapa polutan seperti SO2 dan CO, secara keseluruhan kualitas udara masih menjadi masalah serius yang perlu ditangani.
    """)

if __name__ == "__main__":
    main()