# 💻 Makine Öğrenmesi Tabanlı Akıllı Bilgisayar Öneren Web Sistemi

Bu proje, Bursa Uludağ Üniversitesi Bilgisayar Mühendisliği Bölümü **Python Programlamaya Giriş** dersi kapsamında geliştirilmiş, İçerik Tabanlı (Content-Based) bir akıllı bilgisayar öneri motorudur. 

Standart e-ticaret sitelerindeki teknik filtrelemelerin aksine, bu sistem kullanıcının doğrudan kullanım amacına (Yazılım, Oyun, Ofis) ve bütçesine odaklanarak en uygun cihazı **K-Nearest Neighbors (KNN)** makine öğrenmesi algoritması ile bulur.

## 🚀 Projenin Amacı ve Özellikleri

* **Teknik Karmaşayı Giderir:** Kullanıcıyı donanım terimlerine boğmaz. "Oyun oynamak istiyorum ve bütçem 40.000 TL" diyen bir kullanıcıya arka planda en uygun matematiksel eşleşmeyi sunar.
* **Makine Öğrenmesi (KNN) Entegrasyonu:** Kullanıcının seçimi, çok boyutlu uzayda (CPU, GPU, RAM, Ağırlık) bir "İdeal Hedef Vektörüne" dönüştürülür. Algoritma, bu hedefe Öklid mesafesi olarak en yakın cihazları listeler.
* **Hata Toleranslı Mimari (Fallback):** Proje, önerilen cihazların güncel fiyatlarını çekmek üzere tasarlanmıştır. Canlı veri çekilemediği durumlarda (internet kesintisi vb.) sistem çökmez, yedek veri seti üzerinden çalışmaya devam eder.
* **Dinamik Uygunluk Skoru:** Sonuçlar sıradan bir liste olarak değil, "Bu cihaz sizin için %94 uyumludur" şeklinde şeffaf bir metrik ile sunulur.

## 🛠️ Kullanılan Teknolojiler

* **Dil:** Python 3.x
* **Veri İşleme:** `pandas`, `numpy`
* **Makine Öğrenmesi:** `scikit-learn` (`NearestNeighbors`, `MinMaxScaler`)
* **Veri Çekme (Opsiyonel):** `BeautifulSoup4`

## 📁 Proje Yapısı

\`\`\`text
akilli-bilgisayar-oneri-sistemi/
│
├── main.py                # Ana uygulama ve makine öğrenmesi motoru
├── veri_seti.csv          # Bilgisayar donanım özelliklerini barındıran veri seti
└── README.md              # Proje açıklaması
\`\`\`

## ⚙️ Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. Repoyu bilgisayarınıza klonlayın:
   \`\`\`bash
   git clone https://github.com/azratashan/akilli-bilgisayar-oneri-sistemi.git
   \`\`\`
2. Gerekli kütüphaneleri yükleyin:
   \`\`\`bash
   pip install pandas numpy scikit-learn
   \`\`\`
3. Ana dosyayı çalıştırın:
   \`\`\`bash
   python main.py
   \`\`\`

## 👥 Proje Ekibi

Bu proje aşağıdaki ekip üyeleri tarafından geliştirilmiştir:

* **Kerem Bilgiç** (032490001)
* **Ayşegül Karataş** (032490082)
* **Azra Taşhan** (032490092)

---
*Bursa Uludağ Üniversitesi - 2026*
