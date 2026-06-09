# 💻 Akıllı Bilgisayar Öneri Sistemi (Machine Learning Laptop Recommendation Engine)

Bursa Uludağ Üniversitesi Bilgisayar Mühendisliği Bölümü, **Python Programlamaya Giriş** dersi kapsamında geliştirilmiş; makine öğrenmesi tabanlı, içerik duyarlı (content-based) ve etkileşimli bir karar destek sistemidir.

## 📌 Proje Özeti ve Motivasyon
Standart e-ticaret siteleri kullanıcıya yalnızca "kural tabanlı" (bütçe, marka vb.) filtreleme sunar ve donanımların kullanım senaryosuna uygunluğunu analiz edemez. Bu projenin temel motivasyonu; kullanıcının mesleki ve kişisel amaçlarını (Oyun, Veri Bilimi, Yazılım, Ofis vb.) matematiksel vektörlere dönüştürerek **İçerik Tabanlı Filtreleme (Content-Based Filtering)** yapmaktır. Sistem, pazar donanımlarını analiz eder ve kullanıcı profiline en uygun cihazları yapay zeka algoritmaları ile bularak şeffaf bir "Uygunluk Skoru" eşliğinde sunar.

---

## 🧠 Makine Öğrenmesi Algoritmaları ve Matematiksel Temeller

Sistem, donanımları tarafsız ve bilimsel bir şekilde değerlendirmek için şu modellemeleri kullanır:

1. **Özellik Ölçeklendirme (`sklearn.preprocessing.MinMaxScaler`):**
   Farklı birimlerdeki verilerin (Örn: 32GB RAM ile 7500 MB/s SSD hızı) uzaklık metriklerini bozmasını engellemek için tüm sayısal veriler $0$ ile $1$ aralığına normalize edilmiştir.
2. **Pazar Segmentasyonu (`sklearn.cluster.KMeans`):**
   Veri setindeki bilgisayarların donanım özelliklerine (CPU, GPU, RAM vb.) göre otomatik olarak "Giriş Seviyesi, Orta Segment ve Üst Segment" şeklinde kümelenmesi sağlanmıştır. Arayüzdeki "Performans Sınıfı" seçimi bu algoritmanın çıktılarıyla çalışır.
3. **Öneri Motoru (`sklearn.neighbors.NearestNeighbors`):**
   Kullanıcının ideal vektörüne en yakın cihazları bulmak için K-En Yakın Komşu (KNN) algoritması kullanılmıştır. İki donanım vektörü (hedef $p$ ve cihaz $q$) arasındaki benzerlik, Öklid (Euclidean) mesafesi denklemiyle hesaplanmaktadır: 
   $d(p, q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$

---

## 📂 Modüler Dosya Mimarisi ve Fonksiyon Detayları

Proje "Separation of Concerns" (Sorumlulukların Ayrılması) prensibine göre katmanlara ayrılmış ve düzenli bir klasör yapısında toplanmıştır:

* **`src/create_data.py` (Veri Üretim Katmanı):** Güncel pazar dinamiklerini içeren sentetik donanım verilerinin algoritmik olarak üretildiği modüldür.
* **`src/ml_pipeline.py` (Algoritma Katmanı):** Sistemin matematiksel karar mekanizmasıdır. Veri ön işleme, K-Means eğitimi ve KNN öneri fonksiyonlarını barındırır.
* **`src/evaluation.py` (Değerlendirme Katmanı):** Makine öğrenmesi modellerinin sağlama ve doğruluk testlerinin (Silhouette Skoru vb.) yapıldığı kalite kontrol modülüdür.
* **`app.py` (Sunum Katmanı):** Kullanıcı deneyiminin (UX) sağlandığı ve verilerin Streamlit & Plotly ile görselleştirildiği ana arayüz dosyasıdır.

---

## 👥 Ekip Üyeleri ve Görev Dağılımı

| İsim Soyisim | Üstlenilen Sorumluluklar | İlgili Dosyalar |
| :--- | :--- | :--- |
| **Azra Taşhan** | Güncel piyasa donanımlarını simüle eden sentetik veri setinin algoritmik üretimi, özellik mühendisliği süreçleri, modellerin performans/sağlama testleri ve versiyon kontrol (Git/GitHub) yöneticiliği. | `src/create_data.py`, `src/evaluation.py`, `data/laptop_dataset_2025_2026_2.csv` |
| **Kerem Bilgiç** | Makine öğrenmesi (ML) mimarisinin kurulması, K-Means ve KNN algoritmalarının matematiksel modellenmesi, model eğitimi ve öneri motoru fonksiyonlarının geliştirilmesi. | `src/ml_pipeline.py` |
| **Ayşegül Karataş** | Streamlit altyapısı ile etkileşimli web arayüzü (UI) tasarımı, kullanıcı deneyimi (UX) süreçleri, Plotly veri görselleştirmeleri (Radar/Scatter) ve sistemin uçtan uca senaryo testleri. | `app.py` |

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

* **Veri Seti Linki:** Projede kullanılan ve güncel donanım özelliklerini içeren sentetik veri setine doğrudan bu bağlantıdan ulaşabilirsiniz: [laptop_dataset_2025_2026_2.csv](./data/laptop_dataset_2025_2026_2.csv)

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
