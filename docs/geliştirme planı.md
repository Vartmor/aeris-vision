# AERIS VISION
## Ürün Gereksinimleri ve Ayrıntılı Geliştirme Planı

**Proje 01 / Belge 03 — Nihai Plan v1.0**

**Tarih:** 20 Kasım 2026  
**Hedef:** 2027 BAYKAR / HAVELSAN teknik staj portföyü  
**Proje türü:** Bilgisayarlı görü, güvenilir yapay zekâ, uzaktan algılama ve yazılım mühendisliği  
**Geliştirme ortamı:** LMDE 7 / Linux, RTX 3050 Ti Laptop GPU (4 GB VRAM), 32 GB RAM  
**Planlanan ilk sürüm:** 12 Aralık 2026  
**Staj başvurusu için hedeflenen hazırlık:** 15 Ocak 2027 öncesi

**Belge statüsü:** Uygulama planı. Bu tarihte eğitilmiş model, ölçülmüş başarı skoru veya tamamlanmış AERIS yazılımı varsayılmamaktadır.

---

# 1. Ürün vizyonu ve nihai karar

### 1.1. AERIS VISION nedir?

AERIS VISION, doğal afetlerden önce ve sonra çekilmiş yüksek çözünürlüklü uydu görüntülerini kullanarak binaların konumlarını ve görüntüde gözlemlenebilen hasar seviyelerini tahmin eden, sonuçlarının güvenilirliğini inceleyen bir yapay zekâ araştırma sistemidir.

Sistem yalnızca hasar maskesi üretmeyecek; tahminlerinin hangi koşullarda bozulduğunu, farklı afetlerde nasıl genellendiğini, hangi örneklerde kararsız kaldığını ve sınırlı donanımda ne kadar kaynak tükettiğini de raporlayacaktır.

**Temel ürün vaadi:**

> Afet görüntülerini otomatik analiz eden, tahminlerini ölçülebilir kanıtlarla destekleyen ve belirsiz durumları insan incelemesine yönlendiren, yeniden üretilebilir bir bina hasarı analizi prototipi.

### 1.2. Araştırma hipotezi

Küçük ve önceden eğitilmiş CNN modelleriyle kurulmuş iki aşamalı bir sistemin, 4 GB VRAM sınırı içinde çalışırken anlamlı bina lokalizasyonu ve hasar sınıflandırması üretebileceği; ancak özellikle görülmemiş afetlerde, nadir hasar sınıflarında ve bozulmuş görüntülerde performans kaybı yaşayabileceği araştırılacaktır.

Bu hipotez doğrulanmış sonuç değildir.

### 1.3. Ürünün üç temel özelliği

1. **Gör:** Afet öncesi/sonrası görüntülerde bina maskesi ve dört seviyeli görünür hasar tahmini üret.
2. **Sorgula:** Sınıf bazlı hataları, görülmemiş afet performansını, görüntü bozulmalarını ve tahmin belirsizliğini ölç.
3. **Kanıtla:** Deneyleri, yazılım testlerini, çalışma ortamını ve çıkarım maliyetini yeniden üretilebilir biçimde belgele.

### 1.4. Ürünün yapmayacakları

AERIS, bir binanın fiziksel olarak güvenli veya kullanıma uygun olduğuna karar vermez. Sahada yapılan yapı mühendisliği incelemesinin yerine geçmez. Gerçek zamanlı afet müdahale altyapısı, otomatik arama-kurtarma görevlendirmesi veya operasyonel hedefleme sistemi değildir.

Görüntüde görünür hasar olmaması, binanın güvenli olduğu anlamına gelmez.

---

# 2. Başarı tanımı ve proje kapsamı

## 2.1. İki ayrı teslim seviyesi

| Seviye | Tanım | Hedef |
|---|---|---|
| M0 — Çalışan çekirdek | Veri hattı, iki aşamalı model, uçtan uca çıkarım, test ve temel metrikler | 3 Aralık 2026 |
| M1 — Portföy sürümü | OOD, güvenilirlik, hata analizi, robustness, performans ölçümü, demo ve dokümantasyon | 12 Aralık 2026 |

**M0 tamamlanmadan M1 özelliklerine zaman harcanmayacak.**

M1 için ONNX ve ileri deneyler planlanmıştır; ancak bunlar çalışır sistemi riske atıyorsa küçültülebilir. Proje, tamamlanmamış özellikleri varmış gibi göstermeyecek.

## 2.2. Öncelik sınıfları

**P0 — Vazgeçilmez**

- Veri erişimi, kullanım koşulları ve manifest
- Dört hasar sınıfının doğru işlenmesi
- Veri sızıntısını önleyen train/validation/test ayrımı
- Bina segmentasyonu
- Ground-truth footprint ile hasar sınıflandırması
- Tahmin edilen footprint ile uçtan uca çıkarım
- Sınıf bazlı F1, confusion matrix ve xView2 uyumlu yerel skor
- En az bir held-out-event değerlendirmesi
- Temel testler, README ve çalışan yerel demo

**P1 — Hedeflenen bilimsel tamamlayıcılar**

- Post-only / pre+post karşılaştırması
- CE / weighted CE deneyi
- Temperature scaling
- Manual-review / abstention
- Görüntü bozulması ve hizalama stres testleri
- CPU/GPU gecikme ölçümleri
- ONNX Runtime
- Model card, failure gallery ve teknik demo videosu

**P2 — Zaman kalırsa**

- Lightweight Siamese mimarisi
- Ordinal-aware loss
- İkinci encoder karşılaştırması
- İkinci OOD afet olayı
- ChangeMamba checkpoint karşılaştırması
- BRIGHT araştırması

**P2, P0 veya P1'in tamamlanmasını geciktirmeyecek.**

## 2.3. Başarı kriterleri

Başarı yalnızca yüksek F1 puanı değildir.

| Boyut | Kabul koşulu |
|---|---|
| Veri | Seçilen bütün örneklerde geçerli eşleşme ve açıklanmış etiket politikası |
| Sızıntı kontrolü | Train, validation, calibration ve test arasında yasaklı grup örtüşmesi sıfır |
| Eğitim | En az bir localization ve bir damage modeli gerçekten eğitilmiş |
| Çıkarım | Etiket olmadan pre/post görüntü çiftinden tahmin üretilebiliyor |
| Değerlendirme | Sınıf F1, localization F1, confusion matrix ve OOD raporu mevcut |
| Güvenilirlik | Kalibrasyon ve seçici tahmin uygulanmışsa sonuçları ve sınırları raporlanmış |
| Performans | En az bir gerçek cihazda gecikme ve bellek ölçülmüş |
| Yazılım | Temel testler geçiyor ve kurulum adımları belgelenmiş |
| Sunum | Çalışan demo, mimari açıklaması, model card ve örnek hata analizi mevcut |

**Sayısal F1 hedefi önceden başarı garantisi olarak konulmayacak.** Başlangıçta ölçülmüş bir baseline bulunmadığından keyfî bir skor eşiği, bilimsel olarak sağlam bir kabul kriteri değildir.

---

# 3. Hedef kullanıcı ve kullanım akışı

## 3.1. Birincil kullanıcı

Uydu görüntülerini inceleyen araştırmacı veya afet görüntüsü analisti.

Kullanıcı, iki görüntüdeki bina konumlarını ve görünür hasar dağılımını incelemek, belirsiz örnekleri görmek ve sonuçları dışa aktarmak ister.

## 3.2. Temel kullanım senaryosu

1. Kullanıcı yerel sistemden bir afet öncesi ve bir afet sonrası RGB görüntüsü seçer.
2. Sistem dosya biçimini, boyutları ve eşleşme varsayımlarını kontrol eder.
3. Bina segmentasyonu gerçekleştirilir.
4. Segmentasyon maskesinden bina adayları oluşturulur.
5. Her bina adayı için pre/post görüntü parçaları çıkarılır.
6. Hasar sınıflandırıcısı dört sınıfa ait skorlar üretir.
7. Kalibrasyon uygulanmışsa güven skorları hesaplanır.
8. Belirlenen koşulları karşılamayan tahminler insan incelemesine yönlendirilir.
9. Kullanıcı görüntüleri, maskeleri, bina tablosunu ve model sınırlamalarını inceler.
10. Tahminler JSON/CSV ve uygun maskelerle dışa aktarılır.

Bu akışta kullanıcıya sunulan bina sayısı, modelin tespit ettiği aday sayısıdır; gerçek bina sayısı olarak garanti edilmez.

---

# 4. Veri mimarisi ve veri sözleşmesi

## 4.1. Birincil veri kaynağı

**xBD / xView2**

Ana görevler:

- Bina lokalizasyonu
- Dört sınıflı görünür bina hasarı sınıflandırması
- Öncesi/sonrası görüntü karşılaştırması
- Afetler arası genelleme

Araştırma belgelerinde tam xBD için 850.736 bina anotasyonu, 19 afet ve 22.068 görüntü bilgisi yer almaktadır. Bu rakamlar tam veri kümesine aittir; indirilecek challenge alt kümesinin kapsamı ayrıca doğrulanacaktır.

İlk sürümde tüm veriyi eğitime sokmak zorunlu değildir. Önce doğrulanmış küçük bir alt küme, ardından donanımın izin verdiği ölçüde genişletilmiş veri kullanılacaktır.

## 4.2. Etiket sözleşmesi

| İç değer | Sınıf |
|---|---|
| 0 | Background |
| 1 | No damage |
| 2 | Minor damage |
| 3 | Major damage |
| 4 | Destroyed |
| 255 | Ignore / geçersiz veya sınıflandırılmamış alan — gerektiğinde |

`ignore` bir hasar sınıfı değildir. Orijinal etiketlerdeki sınıflandırılmamış, eksik veya belirsiz anotasyonların nasıl işlendiği veri kartında açıkça belgelenir.

Orijinal JSON alan adları, geometri türleri ve pre/post poligon ilişkileri ilk veri denetiminde doğrulanmadan sabit varsayım olarak kodlanmayacaktır.

## 4.3. Manifest tasarımı

Her görüntü çifti için en az aşağıdaki alanlar tutulacak:

```text
sample_id
event_id
scene_id
pre_image_path
post_image_path
pre_label_path
post_label_path
width
height
damage_class_counts
source_package
validation_status
```

Ek olarak mümkün olduğunda dosya SHA-256 değerleri ve kaynak sürümü tutulacak.

Manifest, model eğitiminden bağımsız olarak üretilebilir olmalıdır.

### Kabul testleri

- Her pre görüntüsünün beklenen post eşleşmesi bulunmalı.
- Eksik dosyalar sessizce atlanmamalı.
- Geçersiz poligonlar raporlanmalı.
- Görüntü dışına taşan koordinatlar kontrollü işlenmeli.
- Bilinmeyen etiketler otomatik olarak `no damage` yapılmamalı.
- Aynı kaynak sahneye ait kayıtların grup kimliği tutarlı olmalı.

## 4.4. Veri bölme stratejisi

İki ayrı deney protokolü kullanılacak.

**Protokol A — ID / geliştirme değerlendirmesi**

Seçilmiş eğitim afetleri içinde, sahne grupları ayrık kalacak şekilde:

- Train: yaklaşık %70
- Validation: yaklaşık %10
- Calibration: yaklaşık %10
- ID test: yaklaşık %10

Bu oranlar başlangıç planıdır. Kesin oranlar örnek ve sınıf dağılımına göre belirlenir.

**Protokol B — OOD / görülmemiş afet**

En az bir afet olayı tamamen dışarıda tutulur.

Bu olayın hiçbir görüntüsü eğitim, validation, calibration, hiperparametre seçimi veya threshold optimizasyonunda kullanılmaz.

OOD olay seçimi, model sonuçları görülmeden veri kapsamı ve sınıf dağılımı incelenerek yapılır. Olayda bulunmayan hasar sınıfları raporda `N/A` olarak belirtilir; var olmayan sınıflar için anlamlı F1 üretilmiş gibi davranılmaz.

### Sızıntı engelleme kuralları

- Aynı pre/post çift farklı split'lere giremez.
- Aynı kaynak sahneden türeyen patch'ler ayrılmaz.
- Augmentation ile üretilmiş örnekler kaynak örneğin split'ini korur.
- Yakın coğrafi tile örtüşmeleri tespit edilebiliyorsa grup düzeyinde kontrol edilir.
- Split manifesti ve hash değeri ilk deneyden önce dondurulur.

Resmî challenge test bölümü ile bu özel ID/OOD testleri birbirine karıştırılmaz.

---

# 5. Sistem mimarisi

## 5.1. Genel mimari

```text
               xBD DATASET
                    |
          Data Audit / Manifest
                    |
          Leakage-Safe Splits
                    |
          +---------+---------+
          |                   |
     Localization        GT Building
       Training           Polygons
          |                   |
    Building Mask         Oracle Crops
          |                   |
    Instance Extraction       |
          |                   |
          +---------+---------+
                    |
           Damage Classifier
                    |
           Confidence / Logits
                    |
         Calibration / Abstention
                    |
       +------------+------------+
       |            |            |
   Evaluation    Inference      Demo
       |            |            |
   ID / OOD      PyTorch      Streamlit
   Robustness    ONNX opt.    JSON / CSV
       |
   Model Card / Reports
```

## 5.2. Katmanların sorumlulukları

| Modül | Sorumluluk |
|---|---|
| `data` | Dosya okuma, JSON parsing, rasterization, crop üretimi |
| `splits` | Grup bazlı veri bölme ve sızıntı denetimi |
| `models` | Localization ve damage ağları |
| `train` | Eğitim döngüsü, checkpoint, logging |
| `eval` | Metrikler, confusion matrix, benchmark |
| `calibration` | Temperature scaling, confidence analizi |
| `robustness` | Kontrollü görüntü bozulmaları |
| `infer` | Gerçek uçtan uca çıkarım |
| `runtime` | PyTorch / ONNX arka uçları |
| `demo` | Kullanıcı arayüzü |
| `reports` | Deney çıktıları, grafikler, model card |

Bu modüller arayüzden bağımsız çalışmalıdır. Streamlit olmadan CLI üzerinden çıkarım ve değerlendirme yapılabilmelidir.

---

# 6. Model mimarisi ve eğitim stratejisi

## 6.1. Model A — Bina lokalizasyonu

**Başlangıç seçimi:** ResNet18 encoder kullanan hafif U-Net.

**Girdi:** Afet öncesi RGB görüntü patch'i.

**Çıktı:** Bina olasılık maskesi.

**Başlangıç ayarları:**

| Parametre | Başlangıç |
|---|---|
| Crop | 256×256 |
| Batch size | 2 |
| Encoder | ImageNet pretrained ResNet18 |
| Loss | BCE + Dice |
| Optimizer | AdamW |
| Learning rate | 1e-4 başlangıç denemesi |
| Precision | Uygunsa AMP/FP16 |
| Checkpoint | Validation localization F1 |
| Mask threshold | Validation üzerinde seçilecek |

Batch size 4 yalnızca bellek ölçümü uygun çıkarsa denenir. Gerekirse 128×128 patch'e geçilir.

Başlangıç öğrenme oranı ve loss ağırlıkları öneridir; ilk smoke-test sonuçlarına göre değiştirilebilir.

### Eğitim sırası

1. 20–50 örnek üzerinde forward/backward testi.
2. Küçük bir veri kümesini aşırı öğrenme testi.
3. Validation metriği ve maske görsellerinin doğrulanması.
4. Gerçek eğitim alt kümesine geçiş.
5. Checkpoint seçimi.
6. Ayrılmış ID testinde değerlendirme.

Aşırı öğrenme testi, veri/etiket hattının çalıştığını doğrular; modelin genellediğini göstermez.

## 6.2. Model B — Hasar sınıflandırması

**Başlangıç seçimi:** ResNet18 tabanlı dört sınıflı classifier.

İki kontrollü girdi biçimi kullanılacak.

**B0 — Post-only baseline**

- Yalnız afet sonrası RGB crop
- 3 giriş kanalı
- Dört hasar sınıfı

**B1 — Pre/post early-fusion**

- Öncesi RGB + sonrası RGB
- 6 giriş kanalı
- Aynı backbone ailesi ve aynı veri split'i
- İlk convolution katmanının pretrained ağırlıkları kontrollü biçimde uyarlanır

Örneğin altı kanallı ilk katman, pretrained üç kanallı ağırlıkların iki zamana paylaştırılmasıyla başlatılabilir. Bu başlatma yöntemi config ve raporda belirtilir.

Amaç, temporal bilginin katkısını kontrollü biçimde test etmektir. İki modelin parametre sayısı ve hesaplama farkları raporlanacaktır.

### Crop üretimi

- Bina poligonunun bounding box'ı bulunur.
- Bağlam için yaklaşık %20–30 kenar boşluğu eklenir.
- Aynı koordinatlardan pre ve post crop alınır.
- Crop hedef giriş boyutuna ölçeklenir.
- Çok küçük veya bozuk poligonlar belirlenmiş politikaya göre işlenir.
- Geometrik augmentasyon pre/post çiftine aynı şekilde uygulanır.

Başlangıç boyutu 160×160 veya 224×224; batch size 4–8 aralığı yalnızca bellek testine göre seçilecektir.

### Eğitim aşamaları

**B0:** Ground-truth bina poligonlarından çıkarılan post-only crop'larla baseline.

**B1:** Aynı bina örneklerinde pre/post erken füzyon.

**B2:** Aynı mimaride normal CE ve sınıf ağırlıklı CE karşılaştırması.

Sınıf ağırlıkları yalnızca eğitim verisinin frekanslarından hesaplanır. Aşırı büyük ağırlıkların eğitimi kararsızlaştırmaması için sınırlama uygulanabilir.

Focal loss, weighted CE ile aynı anda birçok parametreyi değiştiren bir deney olarak değil, zaman kalırsa ayrı bir alternatif olarak değerlendirilecektir.

## 6.3. Model C — Uçtan uca iki aşamalı sistem

Gerçek çıkarım sırasında ground-truth footprint bulunmayacaktır.

Bu nedenle:

1. Localization modeli bina olasılık maskesi üretir.
2. Validation'da seçilmiş eşik uygulanır.
3. Connected-component yöntemiyle bina adayları çıkarılır.
4. Çok küçük bileşenler önceden belirlenmiş kurallarla işlenir.
5. Adayların pre/post crop'ları hazırlanır.
6. Damage modeli her aday için sınıf tahmini üretir.
7. Bina etiketleri tekrar görüntü koordinatlarına yerleştirilir.

**Kritik risk:** Bitişik binalar tek connected component hâline gelebilir. Bu, örnek sayısını ve hasar etiketlerini bozabilir.

İlk sürümde bu durum hata analiziyle ölçülecek. Watershed veya instance-segmentation gibi daha karmaşık çözüm ancak gerçekten gerekli görülürse eklenecektir.

### Oracle ile end-to-end ayrımı

İki ayrı sonuç tutulacak:

- **Oracle-footprint damage:** Gerçek bina geometrileriyle sınıflandırma.
- **Predicted-footprint end-to-end:** Modelin kendi tespit ettiği bina adaylarıyla sınıflandırma.

İkinci sistemin hataları, birincinin skoruyla gizlenmeyecek.

---

# 7. Değerlendirme ve bilimsel deneyler

## 7.1. Resmî skor yaklaşımı

xView2 puanlama yaklaşımında:

`Total = 0.3 × Localization F1 + 0.7 × Damage F1`

Dört hasar sınıfının F1 değerleri harmonik ortalamayla birleştirilir.

Yerel uygulama, resmî açık kaynak scorer ile doğrulanacaktır. Girdi/çıktı biçimi, background işleme ve epsilon davranışı test edilmeden kendi hesapladığımız sonuçlara “resmî benchmark skoru” denmeyecektir.

Özel split üzerinde resmî kodla hesaplanan skor, **xView2 uyumlu yerel değerlendirme** olarak adlandırılır; challenge leaderboard skoru değildir.

## 7.2. Zorunlu metrikler

**Localization**

- Precision
- Recall
- F1
- IoU
- Gerekirse ayrı instance-level eşleşme ölçümleri

**Damage**

- Her sınıf için precision / recall / F1
- Macro-F1
- Harmonic damage-F1
- Confusion matrix
- Destek sayıları (`support`)
- Ağır hasarın hasarsız sınıfa karıştırılması gibi kritik hata türleri

**OOD**

- Olay bazında sınıf dağılımı
- ID ve OOD sonuçları
- Sınıf bazlı performans farkı
- Olay başına örnek sayısı
- Eksik sınıf ve yorumlama sınırları

**Sistem**

- Batch 1 latency
- p50 / p95
- Peak VRAM / RAM
- Parametre sayısı
- Model dosyası boyutu
- Kullanılan cihaz ve yazılım sürümleri

## 7.3. Deney matrisi

| Kod | Deney | Öncelik | Temel soru |
|---|---|---|---|
| E0 | Tiny-subset overfit | P0 | Veri ve eğitim hattı doğru mu? |
| E1 | Post-only vs pre/post | P1 | Öncesi görüntü katkı sağlıyor mu? |
| E2 | CE vs weighted CE | P1 | Nadir hasar sınıfları iyileşiyor mu? |
| E3 | ID vs held-out-event | P0 | Yeni afette model ne kadar bozuluyor? |
| E4 | Oracle vs predicted footprint | P0 | Lokalizasyon hatası ne kadar etkiliyor? |
| E5 | Raw vs calibrated confidence | P1 | Güven tahminleri iyileşiyor mu? |
| E6 | Risk–coverage | P1 | İncelemeye ayırma hata oranını azaltıyor mu? |
| E7 | Blur / JPEG / downsample | P1 | Görüntü kalitesi ne kadar etkiliyor? |
| E8 | ±2/4/8 px shift | P1 | Görüntü hizalaması ne kadar kritik? |
| E9 | Küçük vs orta encoder | P2 | Doğruluk/maliyet dengesi nedir? |
| E10 | PyTorch vs ONNX | P1 | Çıktı tutarlılığı ve gecikme nasıl değişiyor? |

**Deney kuralı:** Her karşılaştırmada yalnızca incelenen faktör değiştirilir. Veri split'i, değerlendirme kodu ve mümkün olduğunca eğitim bütçesi sabit tutulur.

## 7.4. Deney kayıtları

Her eğitim koşusu için aşağıdakiler kaydedilir:

```text
run_id
timestamp
git_commit
config_hash
split_manifest_hash
seed
model_architecture
pretrained_weights
optimizer
learning_rate
batch_size
crop_size
epochs_or_steps
training_duration
peak_vram
checkpoint_path
checkpoint_sha256
validation_metrics
test_metrics
```

En az bir sabit seed ile eksiksiz deney zorunludur. Zaman kalırsa kritik karşılaştırmalar ikinci seed ile tekrarlanır. Tek seed kullanılmışsa sonuçların istatistiksel belirsizliği açıkça belirtilir.

---

# 8. Güvenilirlik, kalibrasyon ve insan incelemesi

## 8.1. Temperature scaling

Hasar modelinin logits çıktıları calibration split üzerinde değerlendirilir.

Tek bir temperature parametresi, calibration verisindeki negative log-likelihood üzerinden optimize edilir.

Sonrasında:

- Ham güven skorları
- Kalibre edilmiş güven skorları
- ECE
- NLL
- Reliability diagram

raporlanır.

Kalibrasyonun mutlaka iyileştirme sağlayacağı varsayılmaz. Sonuç olumsuzsa da kaydedilir.

## 8.2. Abstention / manual review

Başlangıç kararı:

`confidence < τ → manual_review`

Threshold `τ`, yalnızca calibration verisinde önceden tanımlanmış risk–coverage hedefiyle seçilir.

Test sonuçları görüldükten sonra eşik değiştirilmez.

Arayüzde:

- Tahmin edilen sınıf
- Güven değeri
- İnceleme gerekip gerekmediği
- Dört sınıf olasılığı

gösterilir.

**Önemli sınırlama:** Classifier güveni, tamamen kaçırılmış binaları tespit edemez. Bu nedenle yalnızca classifier kalibrasyonundan bütün sistemin güvenilir olduğu sonucu çıkarılmaz.

Ayrıca manual-review durumuna alınan bina, genel benchmark değerlendirmesinden sessizce çıkarılmaz. Tam kapsamlı model skoru ve selective-classification metrikleri ayrı raporlanır.

## 8.3. Risk–coverage değerlendirmesi

Örnek coverage seviyelerinde kalan tahminlerin hata oranı hesaplanır.

Rapor şunları içerir:

- Coverage
- Accepted prediction count
- Rejected prediction count
- Accepted risk
- Full-coverage risk
- Threshold seçimi

Hedef, daha az otomatik karar verildiğinde hangi hataların azaldığını ölçmektir; yalnızca yüksek görünen seçilmiş bir accuracy sayısı üretmek değildir.

---

# 9. Robustness ve hizalama deneyleri

## 9.1. Görüntü bozulmaları

Kontrollü testler:

- Gaussian blur
- JPEG sıkıştırması
- Görüntü küçültme / tekrar büyütme
- İhtiyaç varsa hafif parlaklık/kontrast değişimi

Her bozulma için:

`Baseline F1 → Perturbed F1 → ΔF1`

raporlanır.

Bozulma parametreleri ve rastgele seed kayıt altına alınır.

## 9.2. Misregistration deneyi

Afet sonrası görüntü, öncesi görüntüye göre yapay olarak:

- ±2 piksel
- ±4 piksel
- ±8 piksel

kaydırılır.

Ground-truth etiketler, değerlendirme koordinat sistemi ve sınırda oluşan geçersiz pikseller için açık bir politika korunur.

Amaç gerçek coğrafi kaymanın dağılımını modellemek değil, kontrollü hizalama hatasına duyarlılığı ölçmektir.

## 9.3. Hata galerisi

En az şu örnek türleri seçilir:

- Minor → no damage
- Major → minor
- Destroyed → no damage
- Bitişik binaların birleşmesi
- Kaçırılmış küçük binalar
- Görüntü hizalaması nedeniyle yanlış tahmin
- Düşük güvenli ama doğru tahmin
- Yüksek güvenli ama yanlış tahmin

Örnekler, uygun veri paylaşım hakkı doğrulanmışsa görsel olarak yayımlanır; aksi hâlde yerel raporda tutulur ve kamuya yalnızca izinli/sentetik örnekler çıkarılır.

---

# 10. Teknoloji yığını

| Katman | Tercih |
|---|---|
| Dil | Python |
| ML framework | PyTorch |
| Model bileşenleri | torchvision, gerektiğinde segmentation-models-pytorch |
| Veri işleme | NumPy, Pillow, OpenCV |
| Geometri | Shapely / rasterio, ihtiyaç doğrultusunda |
| Metrikler | scikit-learn + resmî xView2 scorer |
| Konfigürasyon | YAML |
| Paketleme | `pyproject.toml` + kilitlenmiş bağımlılıklar |
| Test | pytest |
| CI | GitHub Actions |
| Demo | Streamlit |
| Inference | PyTorch, hedeflenen ONNX Runtime |
| Sürüm kontrolü | Git / GitHub |

Sürüm numaraları ilk ortam smoke-test'i geçtikten sonra sabitlenecek. PyTorch CUDA wheel'i, NVIDIA driver ve ONNX Runtime sürümleri birlikte doğrulanacak.

Docker, ilk günün zorunlu bağımlılığı değildir. Temel ortam ve CPU kurulumu çalıştıktan sonra eklenebilir.

---

# 11. Repo ve yazılım sözleşmeleri

## 11.1. Hedef repo yapısı

```text
aeris-vision/
├── README.md
├── pyproject.toml
├── configs/
│   ├── data.yaml
│   ├── localization.yaml
│   ├── damage_post.yaml
│   ├── damage_temporal.yaml
│   └── inference.yaml
├── src/
│   └── aeris/
│       ├── cli.py
│       ├── data/
│       ├── splits/
│       ├── models/
│       ├── train/
│       ├── eval/
│       ├── calibration/
│       ├── robustness/
│       ├── infer/
│       ├── runtime/
│       └── demo/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── scripts/
├── docs/
│   ├── architecture.md
│   ├── data_card.md
│   ├── model_card.md
│   ├── evaluation_protocol.md
│   └── limitations.md
├── reports/
├── assets/
└── .github/
    └── workflows/
        └── tests.yml
```

`data/raw`, büyük checkpoint dosyaları, yerel cache ve kısıtlı görseller Git deposuna alınmaz.

## 11.2. Hedef CLI

Aşağıdaki komutlar tasarlanacak arayüzdür; henüz uygulanmış komutlar değildir.

```bash
python -m aeris data audit --config configs/data.yaml

python -m aeris split make --config configs/data.yaml

python -m aeris train --task localization \
  --config configs/localization.yaml

python -m aeris train --task damage \
  --config configs/damage_post.yaml

python -m aeris evaluate --config configs/inference.yaml

python -m aeris calibrate --config configs/inference.yaml

python -m aeris infer \
  --pre path/to/pre.png \
  --post path/to/post.png \
  --output outputs/sample/

python -m aeris benchmark --device cpu
```

Komutların gerçek adları geliştirme sırasında sadeleştirilebilir; temel sözleşme, eğitim ve inference'ın arayüzden bağımsız çalışmasıdır.

## 11.3. Tahmin çıktısı

Örnek JSON şeması:

```json
{
  "sample_id": "sample_001",
  "model_version": "aeris-v0.1",
  "coordinate_system": "image_pixels",
  "buildings": [
    {
      "building_id": "b001",
      "bbox_xyxy": [100, 120, 150, 180],
      "predicted_class": "major_damage",
      "confidence": 0.71,
      "class_probabilities": {
        "no_damage": 0.05,
        "minor_damage": 0.17,
        "major_damage": 0.71,
        "destroyed": 0.07
      },
      "manual_review": true
    }
  ]
}
```

Bu yalnızca **şema örneğidir**, gerçek model tahmini değildir.

Coğrafi koordinatlar, PNG dosyalarından kendiliğinden çıkarılmış gibi sunulmaz. Georeferencing metadata mevcut ve doğrulanmışsa ayrı bir dönüştürme modülü kullanılabilir.

---

# 12. Test stratejisi ve kalite güvencesi

## 12.1. Unit testler

En az şu davranışlar test edilecek:

1. Etiket normalizasyonu
2. Bilinmeyen sınıf işleme
3. Pre/post eşleşme kontrolü
4. Poligon rasterization
5. Boş/bozuk geometri
6. Crop koordinatları
7. Görüntü sınırına taşan crop
8. Split group overlap kontrolü
9. Confusion matrix hesabı
10. Harmonic F1 ve sıfır sınıf davranışı
11. Temperature serialization
12. Manual-review threshold
13. Inference output shape
14. JSON export
15. CPU smoke inference

## 12.2. Integration testler

- Sentetik veri üzerinde manifest → dataset → model → prediction
- Eğitim checkpoint'i kaydetme/yükleme
- Aynı config ile değerlendirme
- CPU inference
- ONNX uygulanmışsa PyTorch–ONNX çıktı karşılaştırması

## 12.3. CI politikası

GitHub Actions üzerinde GPU veya xBD indirmesi gerektirmeyen testler çalıştırılır.

CI'da gerçek veri bulunmaz.

GPU eğitim testleri yerel ortamda yapılır ve sonuçları ayrı kaydedilir.

## 12.4. Reproducibility sınırı

Seed, config, split hash, sürümler ve checkpoint kaydedilir. Farklı CUDA kernel'leri veya donanımlarda mutlak bit-düzeyi eşitlik garanti edilmez.

Amaç sonuçların bilimsel olarak izlenebilir ve makul koşullarda tekrar üretilebilir olmasıdır.

---

# 13. Inference optimizasyonu

## 13.1. Öncelik

Önce doğru PyTorch inference, sonra optimizasyon.

ONNX aktarımı, yanlış veya eksik modeli hızlandırmak için yapılmayacak.

## 13.2. Benchmark protokolü

Her uygun runtime için:

- Batch size 1
- Sabit giriş boyutu
- 20–50 warm-up
- En az 200 ölçüm
- p50 ve p95 latency
- Peak RAM / VRAM
- Model boyutu
- PyTorch–ONNX çıktı farkı

CUDA ölçümünde asenkron çalışma nedeniyle gerekli senkronizasyon uygulanır.

Ayrı iki süre raporlanır:

1. Model-only inference
2. Preprocessing + localization + postprocessing + damage classification içeren end-to-end süre

ONNX CPU hızlanması garanti değildir. Sonuçlar ölçülmeden hızlanma yüzdesi yazılmaz.

## 13.3. Fallback

ONNX export veya CUDA Execution Provider başarısız olursa:

- PyTorch CPU inference korunur.
- Hata ve bağımlılık problemi belgelenir.
- ONNX özelliği tamamlanmamış olarak işaretlenir.
- Projenin bilimsel çekirdeği riske atılmaz.

---

# 14. Demo tasarımı

## 14.1. Ekran 1 — Image comparison

- Pre-disaster görüntü
- Post-disaster görüntü
- Yan yana karşılaştırma
- Eşleşme ve görüntü boyutu bilgisi

## 14.2. Ekran 2 — Building damage overlay

- Tahmin edilen bina maskeleri
- Dört hasar sınıfı için renkler
- Background'ın ayrı tutulması
- İnceleme gerektiren binaların görsel işareti

## 14.3. Ekran 3 — Building inspector

Bir bina seçildiğinde:

- Bina ID
- Crop görüntüsü
- Dört sınıf olasılığı
- Tahmin
- Güven
- Manual-review durumu

## 14.4. Ekran 4 — Evaluation dashboard

Etiketli test verisi mevcutsa:

- Localization F1
- Damage F1
- Confusion matrix
- OOD karşılaştırması
- Risk–coverage grafiği
- Örnek hatalar

Etiketsiz kullanıcı görüntüsünde ground-truth gerektiren metrikler gösterilmez.

## 14.5. Arayüz ilkeleri

- Varsayılan çalışma yerel olacak.
- Kullanıcı görüntüleri dış sunucuya gönderilmeyecek.
- Sonuçlar araştırma tahmini olarak etiketlenecek.
- Lisanslı xBD görselleri izinsiz public demo içine gömülmeyecek.
- Gerçek tahminler ile örnek/sentetik veriler açıkça ayrılacak.

---

# 15. Ayrıntılı geliştirme takvimi

## 15.1. Planlama varsayımı

**Takvim:** 21 Kasım – 12 Aralık 2026  
**Toplam:** 22 takvim günü  
**Net çalışma hedefi:** Yaklaşık 60–80 saat

Bu tahmin, ders yükü yanında ortalama yaklaşık 3 saatlik günlük net çalışmaya dayanır. Bazı günler eğitim işleri arka planda çalışabilir.

Gerçek çalışma süresi bunun altında kalırsa P1 özellikleri azaltılır. Süre tahmini, eğitimin kaç saat süreceğine ilişkin garanti değildir.

## Faz 1 — Veri ve temel altyapı

### Gün 1 — 21 Kasım
**Ortam doğrulama**

- GitHub deposunu oluştur.
- Python ortamını kur.
- NVIDIA driver ve PyTorch CUDA erişimini test et.
- 4 GB VRAM kullanımını ölç.
- CPU fallback'i kontrol et.
- Dataset kullanım koşullarını incele.

**Teslim:** Çalışan ortam, ilk commit, `environment.md`.

### Gün 2 — 22 Kasım
**Veri erişimi ve ilk inceleme**

- xBD erişimini doğrula.
- Küçük bir veri alt kümesini indir.
- Dosya ağacını incele.
- Pre/post eşleşmelerini doğrula.
- Etiket örneklerini görselleştir.

**Teslim:** Veri erişim raporu ve ilk örnek galerisi.

### Gün 3 — 23 Kasım
**Parser ve manifest**

- JSON parser.
- Polygon → mask dönüşümü.
- Dört sınıf eşlemesi.
- Bozuk örnek denetimi.
- Manifest generator.
- İlk veri testleri.

**Teslim:** Doğrulanmış veri okuma hattı.

### Gün 4 — 24 Kasım
**Split ve deney protokolü**

- Event ve scene gruplarını belirle.
- ID train/val/cal/test oluştur.
- OOD event seç.
- Split leakage testini çalıştır.
- Split manifestini dondur.

**Teslim:** `splits.json`, split hash, evaluation protocol.

**Kapı G0:** Veri ve lisans durumu açık, parser çalışıyor, split denetlenmiş olmalı.

## Faz 2 — Model geliştirme

### Gün 5 — 25 Kasım
**Localization smoke-test**

- Hafif U-Net oluştur.
- BCE + Dice loss.
- 20–50 örnek overfit testi.
- Forward/backward ve VRAM ölçümü.

**Teslim:** Çalışan localization training loop.

### Gün 6 — 26 Kasım
**Localization eğitimi**

- Gerçek alt kümede eğitim.
- Validation maskeleri.
- Threshold seçimi.
- İlk F1/IoU.
- Checkpoint kaydı.

**Teslim:** İlk bina segmentasyon modeli.

### Gün 7 — 27 Kasım
**Post-only damage baseline**

- Ground-truth bina crop dataset'i.
- ResNet18 classifier.
- CE loss.
- İlk eğitim ve confusion matrix.

**Teslim:** Oracle-footprint post-only baseline.

### Gün 8 — 28 Kasım
**Pre/post damage modeli**

- Altı kanallı giriş.
- Pretrained ilk katman uyarlaması.
- Aynı split üzerinde eğitim.
- B0/B1 karşılaştırması.

**Teslim:** Temporal bilgi karşılaştırması.

### Gün 9 — 29 Kasım
**Sınıflandırma değerlendirmesi**

- Sınıf F1.
- Minor/major confusion.
- Başarısız crop örnekleri.
- Checkpoint ve deney kayıtları.

**Teslim:** İlk damage evaluation raporu.

### Gün 10 — 30 Kasım
**Dengesizlik deneyi**

- Train sınıf frekansları.
- Weighted CE.
- Sabit koşullarda kısa kontrollü deney.
- CE ile karşılaştırma.

**Teslim:** Loss ablation raporu.

**Kapı G1–G2:** Eğitim hattı doğrulanmış ve seçilen model 4 GB VRAM içinde kararlı çalışıyor olmalı.

## Faz 3 — Uçtan uca değerlendirme

### Gün 11 — 1 Aralık
**Predicted-footprint pipeline**

- Localization maskesinden aday çıkar.
- Connected components.
- Pre/post crop üret.
- Damage tahminini bina adayına bağla.

**Teslim:** Etiketsiz görüntü çiftinden bina tahminleri.

### Gün 12 — 2 Aralık
**Scorer entegrasyonu**

- Tahminleri piksel maskesine geri yaz.
- xView2 scorer biçimini doğrula.
- Bilinen test örnekleriyle metrik testleri yap.
- Oracle ve predicted ayrımını uygula.

**Teslim:** Tekrarlanabilir end-to-end evaluation.

### Gün 13 — 3 Aralık
**M0 sürümü**

- CLI inference.
- Temel JSON çıktısı.
- ID test.
- Confusion matrix.
- İlk hata galerisi.
- README kurulum.

**Teslim:** Çalışan çekirdek sistem, `v0.1.0-m0`.

**Kapı G3:** Model gerçekten uçtan uca tahmin üretiyor ve değerlendirme protokolü çalışıyor olmalı.

## Faz 4 — Bilimsel güvenilirlik

### Gün 14 — 4 Aralık
**OOD değerlendirmesi**

- Held-out-event inference.
- Sınıf bazlı metrikler.
- ID/OOD farkı.
- Olayın sınıf dağılımı.
- Başarısız örnek incelemesi.

**Teslim:** OOD raporu.

### Gün 15 — 5 Aralık
**Robustness**

- Blur/JPEG/downsample.
- ±2/4/8 px misregistration.
- Kontrollü perturbation pipeline.
- F1 değişimleri.

**Teslim:** Robustness tablosu ve grafikler.

### Gün 16 — 6 Aralık
**Kalibrasyon**

- Calibration logits.
- Temperature scaling.
- ECE/NLL.
- Reliability diagram.

**Teslim:** Calibration raporu.

### Gün 17 — 7 Aralık
**Selective classification**

- Threshold seçimi.
- Manual-review çıktısı.
- Risk–coverage.
- Ret edilen örneklerin analizi.

**Teslim:** Belirsizlik ve inceleme raporu.

**Kapı G4:** Güvenilirlik deneyleri tamamlanmış veya neden yapılamadığı açıkça belgelenmiş olmalı.

## Faz 5 — Ürünleştirme

### Gün 18 — 8 Aralık
**ONNX denemesi**

- Model export.
- Output consistency.
- CPU Runtime.
- CUDA Runtime uygunsa test.

**Teslim:** ONNX modeli veya belgelenmiş fallback.

### Gün 19 — 9 Aralık
**Performans benchmark'ı**

- Warm-up.
- 200 inference.
- p50/p95.
- Peak RAM/VRAM.
- Model-only ve end-to-end ölçüm.

**Teslim:** Inference benchmark raporu.

### Gün 20 — 10 Aralık
**Demo**

- Streamlit arayüzü.
- Pre/post karşılaştırma.
- Bina overlay.
- Building inspector.
- Manual-review göstergesi.

**Teslim:** Yerel çalışan demo.

### Gün 21 — 11 Aralık
**Dokümantasyon**

- Model card.
- Data card.
- Limitations.
- Architecture.
- Experiment summary.
- Test/CI kontrolü.
- Teknik demo kaydı.

**Teslim:** Portföy sunum materyalleri.

### Gün 22 — 12 Aralık
**Release ve dondurma**

- Temiz kurulum denemesi.
- README komutlarını yeniden çalıştır.
- Son testler.
- Sonuçları gözden geçir.
- Git tag ve release notes.
- Eksik özellikleri dürüstçe işaretle.

**Teslim:** `AERIS VISION v1.0 — Portfolio Release`

**Kapı G5:** Demo ve belgelenmiş değerlendirme çalışıyor.

**Kapı G6:** Yalnızca bu noktadan sonra P2 araştırmalarına zaman ayrılabilir.

---

# 16. Karar kapıları ve durdurma kuralları

| Kapı | Koşul | Başarısızsa |
|---|---|---|
| G0 | Veri/lisans/parser/split doğrulandı | Eğitime geçme |
| G1 | Tiny overfit ve metrik testleri geçiyor | Model büyütme, veri hattını düzelt |
| G2 | Eğitim 4 GB içinde kararlı | Crop/batch/model küçült |
| G3 | Uçtan uca inference ve skor çalışıyor | Yeni model yerine pipeline onar |
| G4 | OOD ve temel güvenilirlik raporu mevcut | UI/ONNX önceliğini azalt |
| G5 | Demo, test ve doküman tamam | Stretch'i iptal et |
| G6 | M1 tamam, zaman kaldı | Ancak o zaman Siamese/ileri deney |

### Zorunlu kapsam azaltma politikası

- **Veri erişimi gecikirse:** Lisanslı olmayan alternatifleri kullanma; önce erişim sorununu çöz. Çözülemiyorsa sentetik pipeline demonstrasyonu ile gerçek hasar modelini açıkça ayır.
- **GPU OOM olursa:** Batch 1, daha küçük crop, encoder freezing veya daha hafif backbone.
- **Localization başarısızsa:** Oracle-footprint sınıflandırmasını bilimsel baseline olarak koru; end-to-end sistem tamamlandı deme.
- **OOD sınıf kapsamı zayıfsa:** Sonuçları sınıf desteğiyle raporla; tek olaydan geniş genelleme iddiası çıkarma.
- **Takvim sarkarsa:** Önce P2, ardından ONNX ve gelişmiş demo özelliklerini çıkar. Test, dürüst metrikler ve README korunur.

---

# 17. Risk kayıtları

| Risk | Etki | Erken uyarı | Önlem |
|---|---|---|---|
| Dataset erişimi/lisans | Kritik | G0 geçilemiyor | Erişim ve kullanım koşullarını ilk günlerde doğrula |
| Split leakage | Kritik | Aynı scene birden çok split'te | Grup bazlı test ve hash |
| GPU OOM | Yüksek | İlk eğitimde bellek taşması | Küçük model/crop/batch |
| Eğitim çok yavaş | Yüksek | İlk batch throughput düşük | Veri alt kümesi ve deney sayısını azalt |
| Minor/major ayrımı kötü | Yüksek | Confusion matrix yoğun | Hata analizi, weighted loss |
| Localization birleşme hatası | Orta–yüksek | Bitişik binalar tek aday | Instance hata analizi |
| OOD çöküşü | Yüksek | Yeni afette büyük F1 düşüşü | Sonucu gizleme, sınırlama olarak raporla |
| Yanlış kalibrasyon iddiası | Yüksek | ECE kötüleşiyor | Ham ve kalibre metrikleri birlikte yayımla |
| ONNX uyumsuzluğu | Orta | Export/runtime hata veriyor | PyTorch fallback |
| Lisanslı görüntü paylaşımı | Kritik | Public assets içinde ham görüntü | Yayın öncesi lisans denetimi |
| UI zaman tüketimi | Orta | Model raporları eksik | Hazır Streamlit bileşenleri |
| Üç proje takvimi bozulması | Kritik | 12 Aralık sonrası AERIS hâlâ ana iş | Feature freeze ve sonraki projeye geçiş |

Risk olasılıkları ölçülmüş istatistikler değil, proje yönetimi değerlendirmeleridir.

---

# 18. Veri hakları, etik ve güvenlik

## 18.1. Veri paylaşımı

Varsayılan olarak GitHub'da yayımlanmayacaklar:

- Ham xBD görüntü paketleri
- Lisans koşulları doğrulanmamış görüntü crop'ları
- Yetkisiz veri aynaları
- İzin verilmemiş demo ekran görüntüleri

Yayımlanabilecekler, ilgili haklar kontrol edilerek:

- Kendi yazdığımız kod
- Konfigürasyonlar
- Veri indirme ve hazırlama talimatları
- Split oluşturma kodu
- Sentetik test verileri
- Metrik ve deney raporları
- Model card
- Uygun lisanslı ağırlıklar

Model ağırlıklarının paylaşımı da kullanılan pretrained ağırlıkların ve veri koşullarının incelenmesine bağlıdır.

## 18.2. Sorumlu kullanım

Arayüzde ve model card'da şu uyarı bulunacak:

> AERIS VISION araştırma amaçlı bir görüntü analizi prototipidir. Tahminler yalnızca uydu görüntülerinde gözlemlenebilen hasar göstergelerine dayanır. Yapısal güvenlik veya sahada müdahale kararı için kullanılmamalıdır.

## 18.3. Yanlış kararların etkisi

Özellikle:

- Gerçekte ağır hasarlı bir binanın hasarsız tahmin edilmesi
- Görünür hasarın yanlış sınıfa atanması
- Binanın hiç tespit edilememesi
- Yüksek güvenli yanlış tahminler

ayrı hata analizi kategorileri olarak değerlendirilecektir.

---

# 19. Nihai portföy teslimleri

## 19.1. Kod ve sistem

- GitHub repository
- Kurulum ve çalıştırma talimatları
- Veri hazırlama kodu
- Eğitim script'leri
- Inference CLI
- Testler
- CI
- Yerel demo
- Uygunsa ONNX export

## 19.2. Bilimsel kanıt

- Dataset ve split raporu
- Localization sonuçları
- Damage confusion matrix
- Oracle vs end-to-end
- ID vs OOD
- Ablation tablosu
- Calibration / risk–coverage
- Robustness sonuçları
- Latency/VRAM raporu
- Failure gallery

## 19.3. Belgeler

- README
- Architecture
- Data card
- Model card
- Evaluation protocol
- Limitations
- Reproduction guide
- Release notes

## 19.4. Teknik sunum

90–120 saniyelik video için önerilen akış:

1. Problem ve araştırma sınırı
2. Pre/post görüntü
3. Bina maskeleri
4. Hasar tahminleri
5. OOD ve hata analizi
6. Manual review
7. Benchmark ve repo
8. Sonuç ve sınırlamalar

Videoda yalnızca gerçekten uygulanmış özellikler gösterilecektir.

---

# 20. 56 günlük genel portföy takvimine entegrasyon

BAYKAR başvuru hedefi nedeniyle AERIS'in bütün portföyü tüketmesine izin verilmeyecek.

| Tarih | Odak | Süre |
|---|---|---|
| 21 Kasım – 12 Aralık | AERIS VISION | 22 gün |
| 13 – 26 Aralık | AEGIS HEALTH için ayrılan ön pencere | 14 gün |
| 27 Aralık – 8 Ocak | SIMLAB AUTONOMY için ayrılan ön pencere | 13 gün |
| 9 – 15 Ocak | Portföy bütünleştirme, CV ve başvuru | 7 gün |

Son iki projenin kapsamı henüz bu belgeyle kesinleştirilmez; kendi araştırmaları ve geliştirme planlarıyla yeniden değerlendirilecektir.

**Başvuru işlemi 15 Ocak'ın son saatlerine bırakılmayacak.** Hedef, materyalleri 12 Ocak civarında hazır etmek ve başvuruyu son tarihten önce göndermektir.

AERIS'te sonradan yapılacak P2 deneyleri, diğer projelerin zaman bütçesini otomatik olarak kullanamaz.

---

# 21. BAYKAR / HAVELSAN için kanıt haritası

| Teknik yetkinlik | AERIS kanıtı |
|---|---|
| Computer vision | Bina segmentasyonu ve hasar sınıflandırması |
| Deep learning | PyTorch eğitim, transfer learning, loss ablation |
| Data engineering | Manifest, parser, validation, split |
| Scientific evaluation | F1, OOD, calibration, robustness |
| Software architecture | Modüler paket, CLI, test, CI |
| Runtime engineering | CPU/GPU/ONNX karşılaştırması |
| Problem solving | Hata analizi ve karar kapıları |
| Responsible AI | Manual review, model card, sınırlamalar |

Bu eşleştirme teknik portföy stratejisidir; herhangi bir kurumun işe alım kararı veya değerlendirme ağırlığı hakkında garanti değildir.

---

# 22. Nihai karar kaydı

| Karar | Seçim | Gerekçe |
|---|---|---|
| Ana veri | xBD/xView2 | Görev ve benchmark uyumu |
| Localization | Hafif U-Net / ResNet18 | 4 GB VRAM fizibilitesi |
| Damage | ResNet18 classifier | Basit ve kontrol edilebilir baseline |
| Ana sistem | İki aşamalı | Modülerlik ve hata ayrıştırma |
| Temporal deney | Post-only vs early fusion | Öncesi görüntünün katkısını ölçme |
| OOD | Held-out-event | Gerçekçi genelleme değerlendirmesi |
| Güvenilirlik | Temperature scaling + abstention | Ölçülebilir seçici tahmin |
| Robustness | Blur/JPEG/shift | Veri kalitesi ve hizalama hassasiyeti |
| Runtime | PyTorch, ardından ONNX | Önce doğruluk, sonra optimizasyon |
| Demo | Streamlit | Hızlı yerel prototipleme |
| Foundation model | MVP dışı | Donanım ve takvim riski |
| SAR/BRIGHT | MVP dışı | Ayrı sensör ve görev kapsamı |
| Release | 12 Aralık | Diğer projelere zaman bırakma |

---

# 23. İlk uygulama oturumu — kesin başlangıç sırası

Geliştirmeye başlandığında ilk oturumda yalnızca aşağıdakiler yapılacak:

1. Linux, Python, NVIDIA driver, CUDA ve PyTorch sürümlerini tespit et.
2. `nvidia-smi` ve basit bir PyTorch tensor işlemiyle GPU'yu doğrula.
3. Depoyu ve izole Python ortamını oluştur.
4. xBD veri erişim ve lisans koşullarını doğrula.
5. Küçük bir pre/post görüntü çifti ve anotasyonunu aç.
6. Etiket sınıflarını, geometriyi ve eşleşmeyi incele.
7. İlk sentetik parser testini yaz.
8. İlk commit'i oluştur.

**İlk gün hedefi model eğitmek değil, veri ve çalışma ortamının gerçekten doğru olduğunu kanıtlamaktır.**

---

# 24. Kaynaklar ve araştırma temeli

Bu ürün planı iki önceki AERIS araştırma belgesinden türetilmiştir:

- **Araştırma 1/2:** AERIS VISION — Bilimsel ve Teknik Ön Araştırma, 20 Kasım 2026.
- **Araştırma 2/2:** AERIS VISION — Literatür, Mimari Kararlar ve Teknik Fizibilite, 20 Kasım 2026.

Başlıca birincil teknik referanslar:

- [xBD makalesi](https://arxiv.org/abs/1911.09296)
- [TorchGeo xBD](https://docs.torchgeo.org/en/stable/api/datasets/xbd.html)
- [xView2 resmî baseline](https://github.com/DIUx-xView/xView2_baseline)
- [xView2 scoring](https://github.com/DIUx-xView/xView2_scoring)
- [Microsoft Siamese BDA](https://github.com/microsoft/building-damage-assessment-cnn-siamese)
- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [ONNX Runtime](https://onnxruntime.ai/docs/)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)

Bağımlılıkların güncel sürümleri, veri indirme koşulları ve lisans ayrıntıları uygulamanın ilk günlerinde yeniden doğrulanacaktır.

---

# 25. Proje tamamlanma tanımı

AERIS VISION ancak şu durumda portföyde **tamamlanmış** olarak sunulacak:

- Gerçek veri üzerinde eğitilmiş model mevcut.
- Model, etiket gerektirmeden görüntü çiftinden tahmin üretiyor.
- Eğitim ve değerlendirme protokolü açıklanmış.
- Veri sızıntısı denetimi yapılmış.
- Sınıf bazlı metrikler ve en az bir OOD değerlendirmesi var.
- Hatalar ve sınırlamalar gizlenmiyor.
- Kod ve kurulum talimatları çalışıyor.
- En az bir gerçek cihazda kaynak kullanımı ölçülmüş.
- Demo, gerçek sistem davranışını gösteriyor.
- Kullanılan veri ve üçüncü taraf kodlar uygun biçimde atıflanmış.

**Nihai mühendislik ilkesi:**

AERIS VISION'ın başarısı, en karmaşık yapay zekâ modelini kullanmak değildir. Başarısı; kısıtlı kaynaklarla çalışan bir sistemi doğru kurmak, hatalarını ölçmek, sınırlarını açıklamak ve başkalarının denetleyebileceği teknik kanıtlar üretmektir.

**Belge sonu — P01 / D03 / v1.0**
