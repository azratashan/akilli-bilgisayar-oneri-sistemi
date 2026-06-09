# 💻 Akıllı Bilgisayar Öneri Sistemi (Machine Learning Laptop Recommendation Engine)

Bursa Uludağ Üniversitesi Bilgisayar Mühendisliği Bölümü, **Python Programlamaya Giriş** dersi kapsamında geliştirilmiş; makine öğrenmesi tabanlı, içerik duyarlı (content-based) ve etkileşimli bir karar destek sistemidir.

## 📌 Proje Özeti ve Motivasyon
Standart e-ticaret siteleri kullanıcıya yalnızca "kural tabanlı" (bütçe, marka vb.) filtreleme sunar ve donanımların kullanım senaryosuna uygunluğunu analiz edemez. Bu projenin temel motivasyonu; kullanıcının mesleki ve kişisel amaçlarını (Oyun, Veri Bilimi, Yazılım, Ofis vb.) matematiksel vektörlere dönüştürerek **İçerik Tabanlı Filtreleme (Content-Based Filtering)** yapmaktır. Sistem, pazar donanımlarını analiz eder ve kullanıcı profiline en uygun cihazları yapay zeka algoritmaları ile bularak şeffaf bir "Uygunluk Skoru" eşliğinde sunar.

---

## 🛠️ Kullanılan Teknolojiler, Veri Yapıları ve Kütüphaneler

Sistemin arka planında yüksek performanslı veri işleme ve makine öğrenmesi hesaplamaları için şu yapılar kullanılmıştır:

* **`pandas.DataFrame` & `pandas.Series`:** 100 satırlık sentetik veri setinin bellekte etiketli bir matris olarak tutulması, özellik mühendisliği (feature engineering) ve donanım filtreleme işlemlerinin vektörel olarak en hızlı şekilde yapılması için kullanılmıştır.
* **`numpy.ndarray`:** Uzaklık hesaplamaları sırasında özelliklerin işlemci (CPU) düzeyinde optimize edilmiş çok boyutlu dizilere dönüştürülmesi için tercih edilmiştir.
* **Sözlükler (Dictionaries - `dict`):** Kullanıcının "Oyun", "Ofis" gibi soyut seçimlerini, donanım ağırlıklarını içeren (Örn: `{'CPU_Score': 0.9, 'GPU_Score': 0.9, 'Battery_Life': 0.3}`) ideal matematiksel vektörlere eşlemek (mapping) için kullanılmıştır.

---

## 🧠 Makine Öğrenmesi Algoritmaları ve Matematiksel Temeller

Sistem, donanımları tarafsız ve bilimsel bir şekilde değerlendirmek için şu modellemeleri kullanır:

1. **Özellik Ölçeklendirme (`sklearn.preprocessing.MinMaxScaler`):**
   Farklı birimlerdeki verilerin (Örn: 32GB RAM ile 7500 MB/s SSD hızı) uzaklık metriklerini bozmasını engellemek için tüm sayısal veriler $[0, 1]$ aralığına normalize edilmiştir.
2. **Pazar Segmentasyonu (`sklearn.cluster.KMeans`):**
   Veri setindeki bilgisayarların özelliklerine göre (Fiyat, Performans) otomatik olarak "Giriş Seviyesi, Fiyat/Performans, Premium" şeklinde kümelenmesi sağlanmıştır.
3. **Öneri Motoru (`sklearn.neighbors.NearestNeighbors`):**
   Kullanıcının ideal vektörüne en yakın cihazları bulmak için K-En Yakın Komşu (KNN) algoritması kullanılmıştır. İki donanım vektörü (hedef $p$ ve cihaz $q$) arasındaki benzerlik, Öklid (Euclidean) mesafesi denklemiyle hesaplanmaktadır: 
   $d(p, q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$

---

## 📂 Modüler Dosya Mimarisi ve Fonksiyon Detayları

Proje "Separation of Concerns" (Sorumlulukların Ayrılması) prensibine göre 4 farklı katmana ayrılmıştır:

### 1. Veri Üretim Katmanı: `create_data.py`
Hazır veri setleri yerine, güncel pazar dinamiklerini (RTX 5000 serisi, Apple M serisi vb.) içeren sentetik verilerin üretildiği modüldür.
* **`generate_data()`:** Numpy ve random kütüphanelerini kullanarak mantıksal sınırlar çerçevesinde donanım spesifikasyonları üretir ve `laptop_dataset_2025_2026.csv` dosyasına kaydeder.

### 2. Algoritma Katmanı: `ml_pipeline.py`
Sistemin matematiksel karar mekanizmasıdır.
* **`preprocess_data(df)`:** Veri setindeki gereksiz sütunları düşürür, kategorik verileri (Marka, İşletim Sistemi) sayısala çevirir (Encoding) ve `MinMaxScaler` uygular.
* **`apply_hard_constraints(df, min_ram, max_price, os_choice)`:** KNN algoritmasına girmeden önce, kullanıcının "kesin" taleplerini (Sert Kısıtlar) karşılamayan cihazları veri havuzundan eler.
* **`train_kmeans(df, n_clusters)`:** Veri setini eğiterek etiketlerini döndürür.
* **`recommend(user_profile, filtered_df, top_n)`:** Kullanıcı profiline karşılık gelen ideal vektörü oluşturur. `NearestNeighbors.fit()` ile modeli eğitir, `kneighbors()` metodu ile en yakın cihazları ve uzaklık skorlarını hesaplayıp yüzdeye dönüştürerek döndürür.

### 3. Değerlendirme Katmanı: `evaluation.py`
Makine öğrenmesi modellerinin sağlama ve doğruluk testlerinin yapıldığı kalite kontrol modülüdür.
* **`evaluate_clusters()`:** K-Means kümeleme kalitesini `metrics.silhouette_score` ve `metrics.davies_bouldin_score` kullanarak matematiksel olarak ispatlar. Uygulamamızda bu metrikler test edilmiş ve cihazların doğru segmentlere ayrıldığı kanıtlanmıştır.

### 4. Sunum Katmanı (Web UI): `app.py`
Kullanıcı deneyiminin (UX) sağlandığı ve verilerin görselleştirildiği ana çalışma dosyasıdır.
* **`@st.cache_data` (Streamlit Caching):** Veri setinin uygulamanın her etkileşiminde baştan yüklenmesini önler, RAM üzerinde önbellekleme yaparak gecikmeyi sıfıra indirir.
* **Kullanıcı Girdileri:** `st.sidebar`, `st.slider`, `st.selectbox` yapılarıyla dinamik parametreleri anlık olarak `ml_pipeline.py` katmanına aktarır.
* **Görselleştirme (`plotly.express`):** Tavsiye edilen ilk 3 bilgisayarın kıyaslamalarını **Radar Grafikleri** ile, cihazların pazar konumunu ise **Scatter (Dağılım)** grafikleri ile dinamik olarak çizer.

---

## 👥 Ekip Üyeleri ve Görev Dağılımı

| İsim Soyisim | Üstlenilen Sorumluluklar | İlgili Dosyalar |
| :--- | :--- | :--- |
| **Azra Taşhan** | Sistem mimarisi tasarımı, K-Means ve KNN algoritmalarının entegrasyonu, modelin doğruluk testleri ve versiyon kontrol (Git/GitHub) yöneticiliği. | `ml_pipeline.py`, `evaluation.py` |
| **Kerem Bilgiç** | Güncel piyasa donanımlarını simüle eden sentetik veri setinin algoritmik üretimi, özellik mühendisliği ve normalizasyon süreçleri. | `create_data.py`, `*.csv` |
| **Ayşegül Karataş** | Streamlit altyapısı ile web arayüzü (UI) tasarımı, Plotly görselleştirmeleri (Radar/Scatter) ve sistemin uçtan uca senaryo testleri. | `app.py` |

---

## 🚀 Kurulum ve Çalıştırma Yönergeleri

Projeyi kendi bilgisayarınızda yerel ortamda (Localhost) test etmek için terminal üzerinden şu adımları izleyiniz:

**1. Gerekli Kütüphaneleri Yükleyin:**
```bash
pip install -r requirements.txt
```

**2. Arayüzü Başlatın:** 
```bash
streamlit run app.py
```


