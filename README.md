# Omurga X-Ray Analiz Sistemi

SpineAI WebApp, omurga röntgen görüntüleri üzerinde yapay zeka destekli analiz yapmak için hazırlanmış Flask tabanlı bir web uygulamasıdır. Uygulama iki farklı modeli destekler:

- Çok sınıflı model: `normal`, `kayma`, `skolyoz`
- İki sınıflı skolyoz modeli: `normal`, `skolyoz`

Analiz sonrasında tahmin sonucu, güven oranı, sınıf olasılıkları, Grad-CAM görselleri, analiz geçmişi ve PDF rapor üretimi web arayüzünden görüntülenebilir.

> Bu proje klinik karar destek / araştırma amacıyla hazırlanmıştır. Tek başına tıbbi tanı aracı olarak kullanılmamalıdır. Nihai değerlendirme uzman hekim tarafından yapılmalıdır.

## Özellikler

- Kullanıcı kayıt ve giriş ekranları
- Profil ayarları ve şifre güncelleme
- Profil ekranında şifre göster / gizle kontrolü
- Yeni röntgen analizi oluşturma
- Çok sınıflı model ile `normal`, `kayma`, `skolyoz` sınıflandırması
- İki sınıflı EfficientNetB0 modeli ile skolyoz tespiti
- Analiz sonrası Grad-CAM çıktıları
- Analiz geçmişi ve detay açılımı
- Geçmiş analiz detayında görselleri büyüterek inceleme
- Görseller arasında ileri / geri gezinme
- Modern PDF raporu indirme
- PDF raporda tahmin bilgileri ve Grad-CAM görselleri
- Hasta adı ve not gibi kullanıcı girdilerinde HTML escape ile XSS riskini azaltma

## Klasör Yapısı

```text
SpineAI_WebApp_GitHub/
  README.md
  requirements.txt
  start_webapp.bat
  .gitignore
  webapp/
    app.py
    static/
      index.html
      login.html
      register.html
      app.js
      style.css
  model_egitim/
    model.h5
    Models/
      Models_EfficientNetB0_noKayma/
        efficientnetb0_no_kayma_final.weights.h5
```

Bu klasörde yalnızca WebApp'in çalışması için gerekli dosyalar tutulmuştur. Eğitim veri setleri, eski log dosyaları, geçici dosyalar, `__pycache__`, yüklenen analiz görselleri ve çalışma sırasında oluşan veritabanı dosyası GitHub paketine dahil edilmemiştir.

## Kullanılan Modeller

### 1. Çok Sınıflı Model

Dosya yolu:

```text
model_egitim/model.h5
```

Bu model üç sınıflı analiz için kullanılır:

- `normal`
- `kayma`
- `skolyoz`

Web arayüzünde genel omurga analizi seçildiğinde bu model çalışır.

### 2. İki Sınıflı Skolyoz Modeli

Dosya yolu:

```text
model_egitim/Models/Models_EfficientNetB0_noKayma/efficientnetb0_no_kayma_final.weights.h5
```

Bu model sadece iki sınıf içerir:

- `normal`
- `skolyoz`

Web arayüzünde "Skolyoz Tespiti" modeli seçildiğinde bu model yüklenir ve tahmin sonucu iki sınıf üzerinden hesaplanır.

## Çalışma Mantığı

Uygulamanın ana sunucu dosyası `webapp/app.py` dosyasıdır. Flask uygulaması başlatıldığında statik HTML, CSS ve JavaScript dosyalarını `webapp/static/` klasöründen sunar.

Genel analiz akışı şu şekildedir:

1. Kullanıcı kayıt olur veya giriş yapar.
2. Yeni analiz ekranından bir röntgen görüntüsü yükler.
3. Kullanıcı hangi modelin kullanılacağını seçer.
4. Sunucu dosya tipini kontrol eder ve görüntüyü geçici olarak kaydeder.
5. Görüntü model girişine uygun hale getirilir:
   - Görüntü okunur.
   - Merkez kırpma yapılır.
   - Görüntü `224x224` boyutuna getirilir.
   - Kontrast iyileştirme için CLAHE uygulanır.
6. Seçilen model belleğe yüklenir. Model daha önce yüklenmişse tekrar yüklenmez, cache üzerinden kullanılır.
7. Model tahmin üretir.
8. Tahmin sonucu, güven oranı ve sınıf olasılıkları hesaplanır.
9. Grad-CAM üretimi yapılır.
10. Orijinal görüntü, işlenmiş görüntü, heatmap ve overlay çıktıları kaydedilir.
11. Analiz kaydı `webapp/database.json` içine yazılır.
12. Kullanıcı analiz sonucunu, geçmiş analizleri, detayları ve PDF raporu arayüzden görüntüler.

`database.json`, `uploads/` ve `gradcam_outputs/` klasörleri çalışma sırasında otomatik oluşur. Bu dosyalar kişisel/veri içerebileceği için `.gitignore` içine alınmıştır.

## Kurulum

Python 3.9 veya 3.10 kullanılması önerilir. TensorFlow kurulumu sistem ve Python sürümüne göre değişebildiği için uyumlu bir Python sürümü seçmek önemlidir.

### Windows

Proje klasörüne girin:

```powershell
cd SpineAI_WebApp_GitHub
```

Sanal ortam oluşturun:

```powershell
py -3.9 -m venv .venv
```

Sanal ortamı aktif edin:

```powershell
.venv\Scripts\activate
```

Bağımlılıkları kurun:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Uygulamayı başlatın:

```powershell
python webapp\app.py
```

Alternatif olarak sanal ortam kurulduktan sonra `start_webapp.bat` çalıştırılabilir.

### macOS / Linux

```bash
cd SpineAI_WebApp_GitHub
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python webapp/app.py
```

Sunucu açıldıktan sonra tarayıcıdan şu adrese gidin:

```text
http://localhost:5000
```

## API Uçları

Uygulamada kullanılan temel API uçları:

```text
POST   /api/auth/register
POST   /api/auth/login
POST   /api/auth/logout
GET    /api/auth/me
POST   /api/auth/update
POST   /api/analyze
GET    /api/history
GET    /api/analysis/<analysis_id>
DELETE /api/analysis/<analysis_id>
GET    /api/analysis/<analysis_id>/report.pdf
GET    /api/stats
GET    /api/gradcam/<filename>
GET    /api/uploads/<filename>
```

## PDF Rapor

Analiz detay ekranından PDF raporu indirilebilir. Raporda şu bilgiler yer alır:

- Hasta adı
- Analiz tarihi
- Kullanılan model tipi
- Tahmin edilen sınıf
- Güven oranı
- Sınıf olasılıkları
- Doktor notu
- Grad-CAM görselleri

PDF rapor, analiz sonucunun paylaşılabilir ve arşivlenebilir bir çıktısını oluşturmak için tasarlanmıştır.

## GitHub'a Yükleme Notları

Model dosyaları yaklaşık 17 MB boyutundadır ve GitHub'ın tek dosya limitinin altındadır. Yine de model dosyaları için Git LFS kullanılması daha düzenli bir yaklaşım olabilir.

GitHub'a yüklenmemesi gereken çalışma zamanı dosyaları `.gitignore` içinde tutulmuştur:

```text
webapp/database.json
webapp/uploads/
webapp/gradcam_outputs/
*.log
__pycache__/
.venv/
```

Bu dosyalar uygulama çalıştıkça yeniden oluşur.

## Güvenlik Notları

- Bu sürüm yerel geliştirme ve demo kullanımına uygundur.
- Flask `secret_key` üretim ortamında güçlü ve gizli bir değerle değiştirilmelidir.
- `database.json` basit yerel kayıt tutma için kullanılır. Gerçek kullanımda PostgreSQL, MySQL veya benzeri bir veritabanı tercih edilmelidir.
- Şifre güncelleme ve kullanıcı yönetimi geliştirilirken parola hashleme, rate limit, e-posta doğrulama ve güvenli parola sıfırlama akışı eklenmelidir.
- Hasta verileri hassas veri kabul edilmelidir. Gerçek hasta verisiyle çalışırken KVKK/HIPAA benzeri mevzuatlara uygun güvenlik önlemleri alınmalıdır.

## Sık Karşılaşılan Sorunlar

### TensorFlow kurulumu hata veriyor

Python sürümünü kontrol edin. Python 3.9 veya 3.10 ile sanal ortam oluşturmak genellikle daha sorunsuzdur.

### Model bulunamadı hatası

Model dosyalarının şu konumlarda olduğundan emin olun:

```text
model_egitim/model.h5
model_egitim/Models/Models_EfficientNetB0_noKayma/efficientnetb0_no_kayma_final.weights.h5
```

### Port 5000 kullanımda

Başka bir Flask sunucusu açık olabilir. Çalışan süreci kapatın veya `webapp/app.py` içinde port değerini değiştirin.

### İlk analiz yavaş çalışıyor

İlk analiz sırasında model belleğe yüklenir. Sonraki analizlerde model cache üzerinden kullanıldığı için daha hızlı yanıt alınır.

## Geliştirme İçin Mantıklı Sonraki Adımlar

- Şifremi unuttum ekranına e-posta ile parola sıfırlama akışı eklenebilir.
- Kullanıcı rolleri eklenerek doktor, admin ve araştırmacı yetkileri ayrılabilir.
- Analiz kayıtları JSON yerine gerçek bir veritabanında tutulabilir.
- Model versiyonlama eklenerek her analizde kullanılan model sürümü saklanabilir.
- PDF rapora klinik yorum alanı ve imza bölümü eklenebilir.
- Üretim ortamı için Gunicorn, Waitress veya Docker desteği hazırlanabilir.
- Hasta arama, filtreleme ve tarih aralığına göre geçmiş analiz raporlama eklenebilir.
