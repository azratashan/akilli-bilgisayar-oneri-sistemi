# 💻 Akıllı Bilgisayar Öneri Sistemi (Machine Learning Laptop Recommendation Engine)

Bursa Uludağ Üniversitesi Bilgisayar Mühendisliği Bölümü, **Python Programlamaya Giriş** dersi kapsamında geliştirilmiş; makine öğrenmesi tabanlı, içerik duyarlı (content-based) ve etkileşimli bir karar destek sistemidir.

## 🎥 Proje Demo Videosu
Projenin lokal ortamda nasıl kurulduğunu, çalıştırıldığını ve arayüzün kullanımını gösteren kısa ekran kaydına buradan ulaşabilirsiniz: 
👉 [YouTube Demo Videosu Linki Buraya Gelecek]

---

## 📌 Proje Özeti ve Motivasyon
Standart e-ticaret siteleri kullanıcıya yalnızca "kural tabanlı" (bütçe, marka vb.) filtreleme sunar ve donanımların kullanım senaryosuna uygunluğunu analiz edemez. Bu projenin temel motivasyonu; kullanıcının mesleki ve kişisel amaçlarını (Oyun, Veri Bilimi, Yazılım, Ofis vb.) matematiksel ağırlıklara dönüştürerek **İçerik Tabanlı Filtreleme (Content-Based Filtering)** yapmaktır. Sistem, pazar donanımlarını analiz eder ve kullanıcı profiline en uygun cihazları yapay zeka ve uzman sistem algoritmaları ile bularak şeffaf bir "Uygunluk Skoru" eşliğinde sunar.

---

## 🧠 Makine Öğrenmesi Algoritmaları ve İşlem Motoru

Sistem, donanımları tarafsız ve bilimsel bir şekilde değerlendirmek için şu modellemeleri kullanır:

1. **Özellik Ölçeklendirme (`sklearn.preprocessing.MinMaxScaler`):**
   Farklı birimlerdeki verilerin (Örn: 32GB RAM ile 7500 MB/s SSD hızı) uzaklık metriklerini bozmasını engellemek için tüm sayısal veriler 0 ile 1 aralığına normalize edilmiştir.
2. **Pazar Segmentasyonu (`sklearn.cluster.KMeans`):**
   Veri setindeki bilgisayarların donanım özelliklerine (CPU, GPU, RAM vb.) ve enerji verimliliklerine göre otomatik olarak "Giriş Seviyesi, Orta Segment ve Üst Segment" şeklinde kümelenmesi sağlanmıştır. Arayüzdeki "Performans Sınıfı" seçimi bu algoritmanın çıktılarıyla çalışır.
3. **Kural Tabanlı Uzman Sistem (Expert System):**
   Kullanıcının seçtiği kullanım senaryosuna (Veri Bilimi, Oyun, Ofis vb.) göre donanım bileşenlerine dinamik ağırlıklar atayan kural tabanlı bir karar motoru kullanılmıştır. Cihazların nihai uygunluk skoru; GPU, CPU, RAM ve Portability (Taşınabilirlik) gibi normalize edilmiş değerlerin sektörel katsayılarla çarpılıp toplanmasıyla hesaplanmaktadır. Ayrıca spesifik Apple ve Windows kasa mimarileri için azalan verim (diminishing returns) metrikleri de algoritmaya dahil edilmiştir.

---

## 📂 Klasör Yapısı ve Modüler Dosya Mimarisi

Proje "Separation of Concerns" (Sorumlulukların Ayrılması) prensibine göre katmanlara ayrılmıştır:

```text
akilli-bilgisayar-oneri-sistemi/
├── data/
│   └── laptop_dataset_2025_2026.csv   # Sentetik donanım veri seti (Örnek Veriler)
├── src/
│   ├── create_data.py                 # Veri üretim algoritması
│   ├── evaluation.py                  # Model doğrulama ve test metrikleri
│   └── ml_pipeline.py                 # Veri işleme, K-Means ve Uzman Sistem motoru
├── app.py                             # Streamlit web arayüzü ana çalışma dosyası
├── requirements.txt                   # Proje bağımlılıkları ve kütüphane sürümleri
└── README.md                          # Proje kurulum ve mimari dokümantasyonu
```

---

## 💻 Donanım ve Yazılım Gereksinimleri

* **İşletim Sistemi:** Windows 10/11, macOS veya Linux tabanlı herhangi bir OS.
* **Yazılım:** Python 3.9 veya daha güncel bir sürüm.
* **Donanım:** Uygulama tarayıcı üzerinde çalıştığı için minimum 4 GB RAM ve standart bir işlemci yeterlidir. Model eğitimi hafif olduğu için ekstra donanımsal grafik işlemci (GPU) zorunluluğu yoktur.

---

## 🛠 Kullanılan Teknolojiler ve Kütüphaneler

Projenin arka planında yüksek performanslı veri işleme ve arayüz yönetimi için şu kütüphaneler kullanılmıştır:

* **`pandas (>=2.0)` & `numpy (>=1.24)`:** Veri setinin okunması, veri manipülasyonu, özellik mühendisliği ve vektörel hesaplamalar için.
* **`scikit-learn (>=1.3)`:** Özellik normalizasyonu (MinMaxScaler) ve cihazları segmentlere ayırmak için (K-Means) kullanılmıştır.
* **`streamlit (>=1.30)`:** Makine öğrenmesi modelinin interaktif, modern bir web arayüzüne (UI) dönüştürülmesi için.
* **`plotly (>=5.18)`:** Veri setindeki cihazların dağılımlarının (Scatter) ve önerilen cihazların çok boyutlu analizlerinin (Radar Chart) interaktif olarak çizdirilmesi için.
* **`joblib (>=1.3)`:** İleriki aşamalarda eğitilmiş modellerin yönetimi için dahil edilmiştir.

---

## 🗄️ Veri Seti ve Altyapı Bilgilendirmesi (Jüri Notu)

Bu proje, akademik bir veri bilimi uygulaması olarak tasarlandığı için geleneksel web uygulamalarından farklı bir altyapı kullanır:

* **Veri Seti (Örnek Veri):** İlişkisel bir SQL veritabanı yerine, sistem `data/laptop_dataset_2025_2026.csv` isimli statik veri kümesinden beslenmektedir. Projede kullanılan güncel donanım özelliklerini içeren veri setine doğrudan bu bağlantıdan ulaşabilirsiniz: [laptop_dataset_2025_2026.csv](./data/laptop_dataset_2025_2026.csv)
* **Veritabanı Dosyası (Dump) & .env:** Sistem canlı veri çekmediği ve veritabanı kullanmadığı için `.env` (çevre değişkenleri) yapılandırmasına veya bir DB Dump dosyasına ihtiyaç duymaz.
* **Eğitilmiş Model Dosyası:** K-Means makine öğrenmesi modeli, `app.py` tetiklendiğinde `ml_pipeline.py` üzerinden canlı (on-the-fly) olarak eğitilmekte ve anlık çalışmaktadır. Bu nedenle dizinde önceden kaydedilmiş bir `.pkl` (pickle) model dosyası bulunmamaktadır.
* **Demo Kullanıcı Bilgileri:** Uygulama açık bir karar destek sistemi olduğundan üyelik (Login/Register) sistemi barındırmaz; bu nedenle demo kullanıcı bilgisi gerekmez.

---

## 🚀 Kurulum ve Çalıştırma Yönergeleri (Çalıştırma Komutları)

Projeyi bilgisayarınızda sorunsuz bir şekilde ayağa kaldırmak için terminal veya komut satırınızda (CMD/PowerShell) sırasıyla şu adımları izleyiniz:

**1. Projeyi Bilgisayarınıza İndirin (Clone)**
```bash
git clone [https://github.com/azratashan/akilli-bilgisayar-oneri-sistemi.git](https://github.com/azratashan/akilli-bilgisayar-oneri-sistemi.git)
cd akilli-bilgisayar-oneri-sistemi
```

**2. Gerekli Kütüphaneleri Yükleyin**
```bash
pip install -r requirements.txt
```

**3. Uygulamayı (Streamlit Arayüzünü) Başlatın**
```bash
streamlit run app.py
```

*(Komutu çalıştırdıktan sonra sistem varsayılan tarayıcınızda `http://localhost:8501` adresinde otomatik olarak açılacaktır.)*

---

## 👥 Ekip Üyeleri ve Görev Dağılımı

| İsim Soyisim | Üstlenilen Sorumluluklar |
| :--- | :--- |
| **Azra Taşhan** | Sentetik veri setinin algoritmik üretimi, özellik mühendisliği (Feature Engineering), model doğrulama (Evaluation) testleri ve GitHub/Versiyon Kontrol yöneticiliği. |
| **Kerem Bilgiç** | Makine öğrenmesi mimarisinin (K-Means) ve kural tabanlı Uzman Sistem (Expert System) algoritmalarının matematiksel modellenmesi, ML Pipeline entegrasyonu. |
| **Ayşegül Karataş** | Streamlit altyapısı ile web arayüzünün (UI) kodlanması, Plotly veri görselleştirmeleri, hata yönetimi (Fallback mekanizmaları) ve kullanıcı deneyimi (UX). |
