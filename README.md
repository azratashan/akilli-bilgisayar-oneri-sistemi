# 💻 Makine Öğrenmesi Tabanlı Akıllı Bilgisayar Öneren Web Sistemi

Bu proje, Bursa Uludağ Üniversitesi Bilgisayar Mühendisliği Bölümü **Python Programlamaya Giriş** dersi kapsamında geliştirilmiş, İçerik Tabanlı (Content-Based) bir akıllı bilgisayar öneri motorudur. 

Standart e-ticaret sitelerindeki teknik filtrelemelerin aksine, bu sistem kullanıcının doğrudan kullanım amacına ve bütçesine odaklanarak en uygun cihazı **K-Nearest Neighbors (KNN)** makine öğrenmesi algoritması ile bulur.

## 🛠️ Prototip ve Başlangıç Kodu Hakkında Not

Bu depoda (repository) yer alan kaynak kodlar ve veri seti, projenin çalışan ilk prototipini (initial working prototype) temsil etmektedir. Projenin genel vizyonu ile bu teknik teslimat arasındaki ilişki şu şekildedir:

* Algoritma Doğrulaması: Paylaşılan app.py dosyası, sistemin çekirdeğini oluşturan KNN algoritmasının ve normalizasyon süreçlerinin doğru çalıştığını kanıtlamak amacıyla hazırlanmış "başlangıç kodu"dur.

* Temsili Veri Seti: Mevcut veri_seti.csv dosyası, sistemin mantıksal akışını test etmek için oluşturulmuş, gerçek değerlerden bağımsız küçük ölçekli bir numune veri setidir. Ana proje sürecinde bu veri seti, geniş kapsamlı ve gerçek zamanlı verilerle genişletilecektir.

* Mimari Temel: Bu sürüm, projenin ilerleyen aşamalarında üzerine inşa edilecek olan kullanıcı arayüzü ve makine öğrenmesi katmanlarının temel iskeletini oluşturmaktadır.


## 🚀 Projenin Amacı ve Özellikleri

 Günümüz teknoloji pazarındaki yoğun teknik terminoloji (işlemci mimarileri, RAM frekansları vb.), teknik bilgi düzeyi düşük kullanıcılar için karar verme sürecini zorlaştırmaktadır. Bu sistem; kullanıcının soyut niyetini matematiksel vektörlere dönüştürerek, veri setindeki cihazlarla arasındaki benzerliği ölçer ve donanım darboğazı yaşatmayacak en yüksek performanslı cihazları şeffaf bir "Uygunluk Skoru" ile sunar.
* Akıllı Filtreleme: İşletim sistemi (Windows/macOS) ve bütçe aralığına göre dinamik daraltma.
* Kategori Bazlı Analiz: Yazılım, Oyun, Tasarım, Ofis ve Günlük kullanım senaryolarına özel hedefleme.
*  Hızlı Karar Destek: Matematiksel benzerlik skorları ile saniyeler içinde en yakın 3 cihaz eşleşmesi. 


## 🛠️ Teknik Metodoloji

Sistem, doğruluğu ve hızı kanıtlanmış şu yöntemleri temel almaktadır:
* K-Nearest Neighbors (KNN): Sistemin çekirdeğini oluşturan makine öğrenmesi modelidir; hedef koordinatlara Öklid mesafesi ile en yakın cihazları tespit eder.
  
    $$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$
  
* Min-Max Normalizasyon: Farklı birimlerdeki verileri $[0, 1]$ aralığına ölçeklendirerek (feature scaling) modelin tarafsız değerlendirme yapmasını sağlar.
  
  $$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$
  
* Sert Filtreleme (Hard Constraint): İşletim sistemi tercihi ve bütçe aralığı için sisteme kurallar atanarak veri setinin kural tabanlı olarak optimize edilmesini sağlar.

## 🏗️ Program Mimarisi

Sistem üç katmanlı (3-Tier) bir yapıda kurgulanmıştır:
* Veri Katmanı: Pandas kullanılarak yönetilen cihaz özellikleri veri seti.
* Mantık Katmanı (ML): Scikit-Learn üzerinden çalışan makine öğrenmesi motoru.
* Sunum Katmanı: Streamlit web framework'ü ile geliştirilmiş dinamik kullanıcı arayüzü.

  
## ⚙️ Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. Repoyu bilgisayarınıza klonlayın:
   \`\`\`bash
   git clone https://github.com/azratashan/akilli-bilgisayar-oneri-sistemi.git
   \`\`\`
2. Gerekli kütüphaneleri yükleyin:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
4. Uygulamayı (Prototip) başlatın: 
   \`\`\`bash
   streamlit run app.py
   \`\`\`

  
## 👥 Proje Ekibi

Bu proje aşağıdaki ekip üyeleri tarafından geliştirilmiştir:

* **Kerem Bilgiç** (032490001)
* **Ayşegül Karataş** (032490082)
* **Azra Taşhan** (032490092)

---
*Bursa Uludağ Üniversitesi - 2026*
