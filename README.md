# 💻 Akıllı Bilgisayar Öneri Sistemi (Machine Learning Laptop Recommendation Engine)

Bursa Uludağ Üniversitesi Bilgisayar Mühendisliği Bölümü, **Python Programlamaya Giriş** dersi kapsamında geliştirilmiş; makine öğrenmesi tabanlı, içerik duyarlı (content-based) ve etkileşimli bir karar destek sistemidir.

## 📌 Proje Özeti ve Motivasyon
Standart e-ticaret siteleri kullanıcıya yalnızca "kural tabanlı" (bütçe, marka vb.) filtreleme sunar ve donanımların kullanım senaryosuna uygunluğunu analiz edemez. Bu projenin temel motivasyonu; kullanıcının mesleki ve kişisel amaçlarını (Oyun, Veri Bilimi, Yazılım, Ofis vb.) matematiksel vektörlere dönüştürerek **İçerik Tabanlı Filtreleme (Content-Based Filtering)** yapmaktır. Sistem, pazar donanımlarını analiz eder ve kullanıcı profiline en uygun cihazları yapay zeka algoritmaları ile bularak şeffaf bir "Uygunluk Skoru" eşliğinde sunar.

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

* **`create_data.py` (Veri Üretim Katmanı):** Güncel pazar dinamiklerini içeren sentetik verilerin üretildiği modüldür.
* **`ml_pipeline.py` (Algoritma Katmanı):** Sistemin matematiksel karar mekanizmasıdır. Veri ön işleme, K-Means eğitimi ve KNN öneri fonksiyonlarını barındırır.
* **`evaluation.py` (Değerlendirme Katmanı):** Makine öğrenmesi modellerinin sağlama ve doğruluk testlerinin (Silhouette Skoru vb.) yapıldığı kalite kontrol modülüdür.
* **`app.py` (Sunum Katmanı):** Kullanıcı deneyiminin (UX) sağlandığı ve verilerin Streamlit & Plotly ile görselleştirildiği ana çalışma dosyasıdır.

---

## 👥 Ekip Üyeleri ve Görev Dağılımı

| İsim Soyisim | Üstlenilen Sorumluluklar | İlgili Dosyalar |
| :--- | :--- | :--- |
| **Azra Taşhan** | Sistem mimarisi tasarımı, K-Means ve KNN algoritmalarının entegrasyonu, modelin doğruluk testleri ve versiyon kontrol (Git/GitHub) yöneticiliği. | `ml_pipeline.py`, `evaluation.py` |
| **Kerem Bilgiç** | Güncel piyasa donanımlarını simüle eden sentetik veri setinin algoritmik üretimi, özellik mühendisliği ve normalizasyon süreçleri. | `create_data.py`, `*.csv` |
| **Ayşegül Karataş** | Streamlit altyapısı ile web arayüzü (UI) tasarımı, Plotly görselleştirmeleri (Radar/Scatter) ve sistemin uçtan uca senaryo testleri. | `app.py` |

---

## 🛠 Kullanılan Teknolojiler ve Kütüphaneler

Projenin arka planında yüksek performanslı veri işleme ve arayüz yönetimi için şu kütüphaneler kullanılmıştır:

* **`pandas` & `numpy`:** Veri setinin okunması, veri manipülasyonu, özellik mühendisliği ve vektörel uzaklık hesaplamaları için.
* **`scikit-learn`:** Cihazları segmentlere ayırmak için K-Means ve kullanıcı tercihine en yakın cihazları bulmak için K-En Yakın Komşu (KNN) algoritmalarının çalıştırılması için.
* **`streamlit`:** Makine öğrenmesi modelinin interaktif bir web arayüzüne (UI) dönüştürülmesi için.
* **`plotly`:** Veri setindeki cihazların dağılımlarının ve önerilen cihazların radar grafiklerinin çizdirilmesi için.

---

## 🗄️ Veri Seti ve Veritabanı Bilgisi

Bu projede geleneksel bir ilişkisel veritabanı (SQL vb.) kullanılmamaktadır. Veriler statik bir veri kümesinden (CSV) okunarak işlenmektedir.

* **Veri Seti Linki:** Projede kullanılan ve 100 satırlık güncel donanım özelliklerini içeren sentetik veri setine doğrudan bu bağlantıdan ulaşabilirsiniz: [laptop_dataset_2025_2026.csv](./laptop_dataset_2025_2026.csv)

---

## 🚀 Kurulum ve Çalıştırma Yönergeleri

Projeyi kendi bilgisayarınızda (lokal ortamda) sorunsuz bir şekilde çalıştırmak için terminal üzerinden şu adımları izleyiniz:

**1. Projeyi Bilgisayarınıza İndirin (Clone)**
```bash
git clone [https://github.com/azratashan/akilli-bilgisayar-oneri-sistemi.git](https://github.com/azratashan/akilli-bilgisayar-oneri-sistemi.git)
cd akilli-bilgisayar-oneri-sistemi
```

**2. Gerekli Kütüphaneleri Yükleyin:**
```bash
pip install -r requirements.txt
```

**3. Arayüzü Başlatın:** 
```bash
streamlit run app.py
```


