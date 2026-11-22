# AERIS VISION
## Ürün Gereksinimleri ve Ayrıntılı Geliştirme Planı

**Proje 01 / Belge 03 — Geliştirme Planı v1.1**

**Revizyon tarihi:** 21 Kasım 2026  
**Hedef:** 2027 BAYKAR / HAVELSAN teknik staj portföyü  
**Proje türü:** Bilgisayarlı görü, güvenilir yapay zekâ, uzaktan algılama ve yazılım mühendisliği  
**Geliştirme ortamı:** LMDE 7 / Linux, RTX 3050 Ti Laptop GPU (4 GB VRAM), 32 GB RAM  
**Planlanan ilk sürüm:** 12 Aralık 2026  
**Staj başvurusu için hedeflenen hazırlık:** 15 Ocak 2027 öncesi

**Belge statüsü.** Geliştirme öncesi uygulama ve değerlendirme planı. Model, metrik ve takvim maddeleri planlanan çalışmayı tanımlar.

**Geçerli sürüm.** Uygulama kararları için v1.1 esas alınır. [v1.0 geliştirme planı](geli%C5%9Ftirme%20plan%C4%B1.md) ve iki araştırma belgesi, önceki teknik değerlendirmelerin kaydıdır.

---

# 1. Ürün vizyonu ve nihai karar

### 1.1. AERIS VISION nedir?

AERIS VISION, doğal afetlerden önce ve sonra çekilmiş yüksek çözünürlüklü uydu görüntülerini kullanarak binaların konumlarını ve görüntüde gözlemlenebilen hasar seviyelerini tahmin eden, sonuçlarının güvenilirliğini inceleyen bir yapay zekâ araştırma sistemidir.

Sistem yalnızca hasar maskesi üretmeyecek; tahminlerinin hangi koşullarda bozulduğunu, farklı afetlerde nasıl genellendiğini, hangi örneklerde kararsız kaldığını ve sınırlı donanımda ne kadar kaynak tükettiğini de raporlayacaktır.

**Temel ürün vaadi:**

> Afet görüntülerini otomatik analiz eden, tahminlerini ölçülebilir kanıtlarla destekleyen ve belirsiz durumları insan incelemesine yönlendiren, yeniden üretilebilir bir bina hasarı analizi prototipi.

### 1.2. Araştırma sorusu

> Eğitimde temsil edilen afetlerden ayrılmış ID kalibrasyon kümesinde seçilen manuel inceleme eşiği, görülmemiş bir afette uygulandığında otomatik kabul oranı ve kabul edilen tahminlerin hata oranı nasıl değişiyor?

Ana deney, gerçek bina sınırlarından çıkarılan örneklerde sınıflandırıcının güven skorunu inceler. Modelin kendi bulduğu adaylarla yapılan ek deney, aynı yaklaşımın uçtan uca davranışını ölçer. İki deneyde de önce kalibrasyon ve eşik seçimi tamamlanır, ardından sabit kurallarla nihai ID ve OOD testleri çalıştırılır. Hafif modelin 4 GB VRAM içindeki çalışma bütçesi ilk gerçek eğitim koşularında ölçülür.

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

| Seviye | Kabul ölçütü | Hedef |
|---|---|---|
| M0 — Çalışan çekirdek | Veri hattı, eğitilmiş iki model, etiketsiz çıkarım komutu, temel testler ve validation üzerinde gerçek/tahmin edilen bina sınırı kıyası | 3 Aralık 2026 |
| M1 — Portföy sürümü | Dondurulmuş sürüm modeliyle ID/OOD sonuçları, bina bazlı hata tablosu, kalibrasyon ve risk–coverage, PyTorch CPU/GPU ölçümü, sade demo ve belgeler | 12 Aralık 2026 |

## 2.2. Öncelik sınıfları

**P0 — Sürüm için gerekli**

- xBD erişimi, kullanım koşulları, veri manifesti ve ayrık sahne/konum grupları
- Afet öncesi görüntüde bina segmentasyonu ve afet sonrası bina parçalarında dört sınıflı ağırlıklı CE modeli
- Görüntü çiftinden etiketsiz CLI çıkarımı, JSON çıktısı ve sade yerel demo
- Gerçek ve tahmin edilen bina sınırlarıyla değerlendirme, xView2 uyumlu yerel skor ve bina bazında hata sayımı
- En az bir görülmemiş afet olayı, sınıf desteği ve sınıf bazlı metrikler
- ID/OOD ham ve kalibre edilmiş ECE/NLL, sabit eşik deneyi ve bina adaylarıyla uçtan uca ek değerlendirme
- PyTorch ile CPU/GPU uçtan uca p50/p95 gecikme ve GPU tepe VRAM ölçümü
- Temel testler, kurulum talimatları, README sonuç tablosu, data card, model card ve örnek hata analizi

**P1 — Takvime bağlı deneyler**

- Dondurulmuş modelde görüntü bulanıklığı, JPEG ve hizalama duyarlılığı
- E1, paylaşımlı Siamese ağda `post/post` ile `pre/post` kıyası
- Aynı eğitim koşullarında düz CE ve ağırlıklı CE kıyası, yeterli süre varsa
- Kısa demo kaydı ve ek hata dilimleri, temel sonuçlar hazırsa

**P2 — Ek çalışmalar**

- PyTorch–ONNX kıyası, yalnızca 11 Aralık tamponu boşsa
- Ordinal-aware loss, quadratic weighted kappa, ek encoder ve ikinci OOD olayı
- ChangeMamba, BRIGHT ve C++ ONNX Runtime aracı, sonraki sürümler için

## 2.3. Başarı kriterleri

| Boyut | Kabul koşulu |
|---|---|
| Veri | Geçerli görüntü çiftleri, açıklanmış etiket politikası ve eğitim/validation/calibration/test grupları arasında sıfır yasaklı örtüşme |
| Eğitim | İki checkpoint gerçekten eğitilmiş ve validation metrikleriyle seçilmiş |
| Çıkarım | Etiketsiz pre/post görüntü çiftinden CLI ve yerel demo üzerinden sonuç üretilebiliyor |
| Değerlendirme | Nihai ID/OOD sınıf F1, localization F1, confusion matrix ve bina bazlı hata tablosu mevcut |
| Güvenilirlik | Kalibrasyon sonucu, iki sabit eşik ve ID/OOD risk–coverage raporlanmış |
| Performans | PyTorch CPU ve GPU’da uçtan uca p50/p95 ve GPU’da tepe VRAM ölçülmüş |
| Yazılım ve sunum | Temel testler ve temiz kurulum geçiyor, yeniden üretim adımları ve sınırlamalar belgelenmiş |

Performans hedefleri, ilk çalışan modelin ölçümleri üzerinden yorumlanır. Teslim ölçütü, protokolün uygulanması ve sonuçların açıklanmasıdır.

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
5. Her bina adayı için afet sonrası görüntü parçası çıkarılır.
6. Hasar sınıflandırıcısı dört sınıfa ait skorlar üretir.
7. M1’de sabit sıcaklık katsayısıyla adayın güven skoru hesaplanır.
8. Güven skoru aday eşiğinin altındaysa tahmin insan incelemesine yönlendirilir.
9. Kullanıcı görüntüleri, maskeleri, bina tablosunu ve model sınırlamalarını inceler.
10. Tahminler JSON/CSV ve uygun maskelerle dışa aktarılır.

Bu akışta kullanıcıya sunulan bina sayısı, modelin tespit ettiği aday sayısıdır; gerçek bina sayısı olarak garanti edilmez.

---

# 4. Veri mimarisi ve veri sözleşmesi

## 4.1. Birincil veri kaynağı

İlk sürümün veri kapsamı **xBD / xView2 challenge train paketi**. Gerçek paket için olay, sahne ve sınıf desteği veri denetiminde çıkarılır. İlk paket, eğitim ve validation için dört sınıfı veya seçilen görülmemiş olay için yeterli etiket desteğini sağlamıyorsa Tier 3 paketini kullanım koşullarıyla birlikte G0’da değerlendiririz.

Tam xBD için makalede 850.736 bina anotasyonu, 19 afet ve 22.068 görüntü raporlanmıştır. Sürüm raporu gerçek paket ve seçilen alt küme sayılarını kullanır. Alt küme eğitim performansı ölçülerek büyütülür.

## 4.2. Etiket sözleşmesi

| İç değer | Sınıf |
|---|---|
| 0 | Background |
| 1 | No damage |
| 2 | Minor damage |
| 3 | Major damage |
| 4 | Destroyed |
| 255 | Ignore / geçersiz veya sınıflandırılmamış alan — gerektiğinde |

Geçerli geometrisi olan `un-classified` binalar lokalizasyon ve geometrik eşleştirmeye dâhil edilir. Hasar kaybı, sınıf metrikleri ve hasar kalibrasyonu yalnızca dört hasar sınıfından biriyle etiketlenmiş binaları kullanır. Geçersiz geometri, eksik etiket ve sınıflandırılmamış hasar ayrı sayılır ve veri kartında açıklanır.

Orijinal JSON alan adları, geometri türleri ve pre/post poligon ilişkileri ilk veri denetiminde doğrulanmadan sabit varsayım olarak kodlanmayacaktır.

## 4.3. Manifest tasarımı

Her görüntü çifti için en az aşağıdaki alanlar tutulacak:

```text
sample_id
event_id
event_type
scene_id
spatial_group_id
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

Kaynak sürümü, doğrulanabilen sahne kimliği, `lng_lat` kapsamı ve mevcut çekim metadatası (`gsd`, `off_nadir_angle`, `sun_elevation`) denetlenir. Metadata mevcutsa değeri, eksikse eksiklik durumu kaydedilir. Uygun dosyalar için SHA-256 tutulur.

Manifest, model eğitiminden bağımsız olarak üretilebilir olmalıdır.

### Kabul testleri

- Her pre görüntüsünün beklenen post eşleşmesi bulunmalı.
- Eksik dosyalar sessizce atlanmamalı.
- Geçersiz poligonlar raporlanmalı.
- Görüntü dışına taşan koordinatlar kontrollü işlenmeli.
- Bilinmeyen etiketler otomatik olarak `no damage` yapılmamalı.
- Aynı kaynak sahneye ait kayıtların grup kimliği tutarlı olmalı.

## 4.4. Veri bölme stratejisi

**Protokol A — ID**

- Train (yaklaşık %70), validation (yaklaşık %10), ID kalibrasyon (yaklaşık %10) ve nihai ID test (yaklaşık %10) grupları ayrık tutulur.
- Model, lokalizasyon maskesi eşiği ve aday çıkarma ayarları validation ile seçilir. Train ve validation gruplarında dört hasar sınıfının desteği dondurmadan önce kontrol edilir.
- Sıcaklık katsayısı ve inceleme eşikleri ID kalibrasyon grubunda belirlenir. M0 değerlendirmesi validation kullanır. ID test M1’de model ve eşikler dondurulduktan sonra açılır.
- Oranlar başlangıç hedefidir. Kesin oranlar sahne ve sınıf desteğine göre G0’da belirlenir.

**Protokol B — OOD**

- En az bir afet olayının tamamı nihai OOD testine ayrılır. Olayın görüntüleri model, parametre veya eşik seçiminden ayrı tutulur.
- Tercih, afet türü eğitimde de bulunan ve etiket desteği yeterli bir olaydır. Bu mümkün olmazsa olay ve afet türünün birlikte değiştiği raporda belirtilir.
- Olay, paket kapsamı ve sınıf desteği üzerinden eğitimden önce seçilir. Bir olaydan elde edilen sonuç o olayın koşullarıyla yorumlanır.
- Sınıf desteği her test için yayımlanır. Gerçek örneği bulunmayan sınıflar tanısal sınıf tablosunda `N/A` olur. xView2 uyumlu toplam hesapta scorer’ın dört sınıflı formülü uygulanır ve destek eksikliği yanında belirtilir.

### Sızıntı engelleme kuralları

- Bir görüntü çifti, ondan türeyen parçalar ve augmentasyonları tek bölümde kalır.
- Güvenilir kaynak sahne kimliği olan kayıtlar sahne düzeyinde gruplanır. Konum bilgisi yeterli olduğunda örtüşen veya komşu parçalar, doğrulanan `lng_lat` veya sahne ayak izi üzerinden aynı mekânsal gruba bağlanır.
- Gruplama yöntemi, konum ve sahne bilgisi eksik kayıtların sayısı ve bu kayıtlar için kullanılan yaklaşım veri raporunda belgelenir.
- Sahne, mekânsal grup ve olay sınırları otomatik kontrollerden geçirilir. Bölme manifesti, olay listesi ve hash ilk eğitimden önce dondurulur.
- ID ve OOD, challenge train paketi üzerinde oluşturulan AERIS bölümleridir. Raporlarda resmî challenge test bölümü kendi adıyla anılır.

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

Checkpoint, validation görüntülerinde ölçülen **ikili piksel localization F1** ile seçilir. Maske eşiği ve küçük bileşen kuralları validation ile belirlenir. Bu ayarlar nihai değerlendirmeden önce dondurulur. Başlangıç öğrenme oranı ve loss ağırlıkları smoke-test ölçümlerine göre ayarlanabilir.

### Eğitim sırası

1. 20–50 örnek üzerinde forward/backward testi.
2. Küçük bir veri kümesini aşırı öğrenme testi.
3. Validation metriği ve maske görsellerinin doğrulanması.
4. Gerçek eğitim alt kümesine geçiş.
5. Checkpoint seçimi.
6. Validation’da maskeleri ve hataları incele. Nihai ID/OOD değerlendirmesi M1’in dondurulmuş değerlendirme koşusunda yapılır.

Aşırı öğrenme testi, veri/etiket hattının çalıştığını doğrular; modelin genellediğini göstermez.

## 6.2. Model B — Hasar sınıflandırması

**12 Aralık sürümü.** ImageNet ön eğitimli ResNet18, afet sonrası üç kanallı RGB bina parçasını alır ve dört hasar sınıfı için logits üretir. Eğitimde bina parçaları gerçek poligonlardan çıkarılır. Uçtan uca çıkarımda aynı crop sözleşmesi tahmin edilen bina adaylarına uygulanır.

### Crop üretimi

- Bina poligonunun veya aday maskesinin sınır kutusu alınır ve bağlam için yaklaşık %20–30 kenar boşluğu eklenir.
- Afet sonrası görüntüden crop alınır ve hedef boyuta ölçeklenir. Kenara taşan, çok küçük ve geçersiz kutular için aynı yazılım politikası kullanılır.
- Başlangıçta 160×160 veya 224×224 giriş ve 4–8 batch aralığı denenebilir. Seçim ilk bellek ve throughput ölçümlerine dayanır.
- PNG tekrar çözme işlemi darboğaz çıkarsa crop’lar split bazında önceden çıkarılır ve önbellek, manifest ile crop konfigürasyonuna bağlanır.

### Kayıp ve checkpoint

Varsayılan kayıp sınıf ağırlıklı çapraz entropidir. `n_c`, OOD olayı ayrıldıktan sonra train kümesindeki geçerli eğitim binaları içinde sınıf `c` sayısıdır. `n_max` en çok desteklenen sınıfın sayısıdır.

`w_c = min(4, sqrt(n_max / n_c))`

Sayımlar, crop ve örnek filtreleme kurallarından geçen binalardan augmentasyon öncesinde alınır ve deney kaydına yazılır. Train’de dört sınıfın da desteği G0’da aranır. Ağırlıklı kayıp modelin ham sınıf skorlarını değiştirebilir. Ham güven skorlarının kalibrasyon kalitesi ID/OOD ölçümleriyle belirlenir.

Sınıflandırıcı checkpoint’i, **validation kümesindeki gerçek bina sınırlarından çıkarılan crop’larda dört sınıfın harmonik damage F1’i** ile seçilir. Dört sınıftan herhangi birinin F1’i sıfırsa bu harmonik ortalama sıfırdır. İlk eşitlik ölçütü macro-F1, ikinci eşitlik ölçütü daha erken checkpoint’tir. Bu seçim bina başına bir etiket kullanır. Piksel tabanlı xView2 hasar skoru 7.1’deki değerlendirmede ayrıca hesaplanır.

### E1 ve E2

- **E1.** Aynı paylaşımlı ResNet18 Siamese crop sınıflandırıcısında `post/post` ve `pre/post` kıyaslanır. Veri bölümü, aday crop koordinatı, model başlığı, kayıp ve eğitim bütçesi sabit tutulur. `post/post` için aynı dönüştürülmüş post crop iki kola verilir. `pre/post` çiftinde geometrik augmentasyon iki zamana birlikte uygulanır.
- **E2.** Vakit kalırsa aynı post-only mimaride düz CE ve ağırlıklı CE kıyaslanır.
- E1 ve E2 sonuçları validation üzerinde keşifsel olarak raporlanır. 12 Aralık sürümü, 4 Aralık nihai testi için dondurulan post-only model ve eşikleri kullanır.

## 6.3. Model C — Uçtan uca iki aşamalı sistem

Gerçek çıkarım sırasında ground-truth footprint bulunmayacaktır.

Bu nedenle:

1. Localization modeli bina olasılık maskesi üretir.
2. Validation'da seçilmiş eşik uygulanır.
3. Connected-component yöntemiyle bina adayları çıkarılır.
4. Çok küçük bileşenler önceden belirlenmiş kurallarla işlenir.
5. Her aday için afet sonrası crop hazırlanır.
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

## 7.1. xView2 uyumlu piksel skoru

`Total = 0.3 × Localization F1 + 0.7 × Damage F1`

Dört hasar sınıfının F1 değerleri ve epsilon davranışı resmî scorer’da tanımlandığı biçimde birleştirilir. Sınıflandırılmamış geçerli bina geometrisi localization hedefinde bina olarak korunur. Resmî scorer’a verilen damage hedefinde bu pikseller 0 olarak yazılır ve scorer’ın hasar değerlendirme alanı dışında kalır. İç maskelerde kullanılan `255` değeri dışa aktarmada scorer sözleşmesine dönüştürülür.

İlk gerçek eğitimden önce 1024×1024 boyutlu, beklenen sonucu bilinen sentetik maskelerle biçim ve skor doğrulanır. Aynı ID/OOD görüntülerindeki oracle ve uçtan uca **damage F1**, aynı piksel puanlama koduyla hesaplanır. Oracle koşulu gerçek bina sınırlarını kullanır. Nihai birleşik skor, tahmin edilen lokalizasyon ile hasar maskelerinden hesaplanan **xView2 uyumlu yerel skor** olarak raporlanır.

## 7.2. Zorunlu metrikler

- **Lokalizasyon.** İkili piksel precision, recall, F1 ve IoU.
- **Hasar.** Sınıf başına precision, recall, F1 ve destek, macro-F1, harmonik damage F1 ve confusion matrix. Bina ve piksel düzeyleri tablo başlığında belirtilir.
- **Bina bazında.** Aşağıdaki bire bir eşleştirme, birleşme sayıları ve ağır hasar hata dökümü.
- **ID/OOD.** Olay türü, örnek sayıları, sınıf desteği, eşik grafiği ve hata örnekleri.
- **Sistem.** PyTorch CPU/GPU uçtan uca p50/p95 gecikme ve GPU tepe VRAM. Ölçüm koşulları 13.2’de tanımlanır.

### 7.2.1. Bina eşleştirme ve birleşme kuralları

1. **Eşleştirme.** Görüntü başına, aynı piksel ızgarasındaki aday maskeleri ve etiketli bina maskeleri için `IoU = kesişim alanı / birleşim alanı` hesaplanır. `IoU ≥ 0,5` sağlayan çiftler azalan IoU ile sıralanır. Eşit IoU’da önce gerçek bina kimliği, sonra aday kimliği artan sırada kullanılır. Her iki tarafı da boş olan ilk uygun çift alınarak açgözlü bire bir eşleştirme yapılır.
2. **Birleşme.** Bire bir eşleştirmeden bağımsız olarak, bir aday en az iki etiketli binanın her birinde `kesişim alanı / gerçek bina alanı > 0,5` sağlıyorsa birleşik adaydır. Birleşik aday sayısı ve içindeki benzersiz etiketli bina sayısı ayrı raporlanır.
3. **Ağır hasar tablosu.** Payda, tüm gerçek `major` ve `destroyed` binalardır. Her gerçek bina bir kez sayılır. Birleşik bir adayın içinde bulunanlar önce `birleşik adayda` sütununa yazılır. Kalan eşleşmesiz binalar `bulunamadı` sütununa yazılır. Diğer binalar bire bir eşleştirmedeki tahmin sınıfına göre `no damage`, `minor`, `major` veya `destroyed` sütunlarına girer. Böylece birleşik adayda kalan bir bina hem birleşmede hem kaçırmada sayılmaz.
4. **Bire bir lokalizasyon ve güvenilirlik.** Bire bir recall, eşleşen gerçek binaların tüm etiketli binalara oranıdır. Ağır hasar tablosu, birleşme kaynaklı eşleşme kayıplarını ayrıca gösterir. Aday bazındaki risk hesabı 8.3’te tanımlanır.

Bu eşikler ve öncelik sırası, nihai sonuçlar açılmadan önce değerlendirme konfigürasyonuna yazılır. Birleşme, eşleşemeyen aday ve ağır hasar raporu için sentetik kontrol örnekleri hazırlanır.

## 7.3. Deney matrisi

| Kod | Deney | Öncelik | Temel soru |
|---|---|---|---|
| E0 | Tiny-subset overfit | P0 | Eğitim hattı küçük örnekleri öğrenebiliyor mu? |
| E1 | Aynı Siamese ağda post/post ve pre/post | P1, validation | Öncesi görüntü bu mimaride ne katıyor? |
| E2 | Ağırlıklı CE ve düz CE | P1, validation | Ağırlıklandırma sınıf bazlı sonuçları nasıl değiştiriyor? |
| E3 | ID ve eğitimden ayrılmış olay | P0 | Sınıf performansı yeni olayda nasıl değişiyor? |
| E4 | Oracle ve tahmin edilen bina sınırları | P0 | Hangi hatalar lokalizasyon aşamasıyla ilişkili? |
| E5 | Ham ve kalibre edilmiş güven | P0 | ID ve OOD’da ECE/NLL nasıl değişiyor? |
| E6 | Sabit eşikli risk–coverage | P0 | ID’de belirlenen eşik yeni afette nasıl davranıyor? |
| E7 | Blur ve JPEG | P1 | Görüntü kalitesi performansı nasıl etkiliyor? |
| E8 | ±2/4/8 px shift | P1 | Görüntü hizalaması performansı nasıl etkiliyor? |
| E9 | Küçük ve orta encoder | P2 | Doğruluk ve maliyet dengesi nasıl değişiyor? |
| E10 | PyTorch ve ONNX | P2 | Çıktı ve gecikme nasıl değişiyor? |

Kıyaslarda incelenen faktör dışındaki veri bölümü, değerlendirme kuralları ve mümkün olduğunca eğitim bütçesi sabit tutulur.

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

Model dondurulduğunda kalibrasyon ve test için ham logits, örnek/aday kimliği, değerlendirme grubu, aday geometrisi, etiket ve eşleştirme bilgileri saklanır. Kalibrasyon, eşik taraması ve bootstrap bu kayıtlar üzerinden yeniden eğitim olmadan yapılabilir.

En az bir sabit seed kaydedilir. Yeterli bağımsız grup olduğunda güven aralıkları sahne/mekânsal blok bootstrap’ıyla hesaplanır. Eşli bir kıyasta iki yöntem için aynı bloklar yeniden örneklenir. Aralıklar test örneklerinin değişkenliğini yansıtır. Küçük bir model farkına dayanan iddia için eğitim değişkenliği ikinci seed ile incelenir. Tek seed veya az bağımsız blok kullanılıyorsa bu kapsam raporda belirtilir.

---

# 8. Güvenilirlik, kalibrasyon ve insan incelemesi

## 8.1. Temperature scaling

Dondurulmuş hasar checkpoint’inin, gerçek bina sınırlarından çıkarılan **ID kalibrasyon** örneklerine ait logits değerleri kullanılır. Tek bir pozitif `T`, ağırlıksız negative log-likelihood (NLL) en aza indirilerek bulunur. Aynı `T` ID ve OOD’da ve demodaki aday skorlarında uygulanır. Güven, `max(softmax(logits / T))` olarak hesaplanır.

Gerçek bina sınırı örneklerinde ham ve kalibre edilmiş **ECE ve NLL**, ID ile OOD için ayrı ayrı ve aynı örnekler üzerinde raporlanır. ECE bölme kuralı testten önce sabitlenir. Sıcaklık ölçekleme tahmin edilen sınıfı korur. Dört sınıflı görevde binaların güven sıralaması değişebilir. İnceleme performansı 8.3’teki risk ve kapsama değerleriyle değerlendirilir.

## 8.2. Manuel inceleme eşikleri

Her iki eşik, aynı ID kalibrasyon bölümünde `confidence ≥ τ` ile yaklaşık **%80 kapsama** sağlayan skor üzerinden seçilir. Hedefe en yakın gerçekleşebilir kapsama seçilir. Eşit uzaklıkta daha yüksek kapsama kullanılır ve eşit skorlu örnekler aynı kararı alır. Gerçekleşen ID kalibrasyon kapsamı rapora yazılır.

| Eşik | Seçildiği ID kalibrasyon örnekleri | Uygulama |
|---|---|---|
| `τ_oracle` | Geçerli hasar etiketli gerçek bina crop’ları | Ana deneyde sabit, hem nihai ID hem OOD |
| `τ_candidate` | Uçtan uca hattın ürettiği tüm bina adayları | Ek deneyde sabit, hem nihai ID hem OOD ve 12 Aralık demosu |

Bir adayın güveni `τ_candidate` altındaysa `manual_review` işaretlenir. Her iki eşik ve sıcaklık katsayısı nihai testten önce dondurulur. Tahmin edilen sınıf ve dört sınıf skoru gösterilir. xView2 uyumlu yerel skor, incelemeye ayrılanlar dâhil tam tahmin setini kullanır.

## 8.3. Risk–coverage değerlendirmesi

- **Ana grafik.** Gerçek bina sınırı örnekleri için ID ve OOD risk–coverage eğrileri aynı grafikte gösterilir. Her eğri üzerinde aynı sabit `τ_oracle` eşiğinin gerçekleştiği nokta işaretlenir. Kapsama, kabul edilen örneklerin tüm geçerli hasar örneklerine oranı. Risk, kabul edilen örneklerdeki yanlış sınıf oranı.
- **Uçtan uca ek grafik.** Aynı gösterim `τ_candidate` ile modelin ürettiği adaylarda kullanılır. Kapsamanın paydası tüm üretilen adaylardır. Puanlanabilir kabul edilmiş adaylarda yanlış sınıf ve eşleşmesiz aday hata sayılır. Sınıflandırılmamış bir binayla eşleşen adayın hasar riski hesaplanmaz ve sayısı ayrıca verilir.
- **Bulunmayan binalar.** Aday güveni üretilemeyen gerçek binalar aday kapsamasının dışında kalır. Bire bir eşleşmesiz bina oranı, tüm etiketli binalar üzerinden verilir. Ağır hasar tablosu, birleşme ve diğer kaçırma türlerine ait dökümü içerir.
- **Rapor.** Sabit eşik, kabul/inceleme sayıları, kapsama, kabul edilenlerin riski ve tam kapsama riski her bölüm için kaydedilir. Kabul edilen puanlanabilir örnek sayısı sıfırsa risk `N/A` olarak gösterilir.

---

# 9. Robustness ve hizalama deneyleri

## 9.1. Görüntü bozulmaları

Takvim elverirse Gaussian blur ve JPEG sıkıştırması dondurulmuş sürüm modelinde ölçülür. Parametre seviyeleri ve veri bölümü sonuçlara bakılmadan belirlenir.

Her bozulma için:

`Baseline F1 → Perturbed F1 → ΔF1`

raporlanır.

Bozulma parametreleri ve rastgele seed kayıt altına alınır.

## 9.2. Misregistration deneyi

İsteğe bağlı hizalama deneyinde afet sonrası görüntü, öncesi görüntüye göre yapay olarak

- ±2 piksel
- ±4 piksel
- ±8 piksel

olacak şekilde kaydırılır.

Ground-truth etiketler, değerlendirme koordinat sistemi ve sınırda oluşan geçersiz pikseller için açık bir politika korunur.

Amaç gerçek coğrafi kaymanın dağılımını modellemek değil, kontrollü hizalama hatasına duyarlılığı ölçmektir.

## 9.3. Hata galerisi

En az şu örnek türleri seçilir:

- Minor → no damage
- Major → minor
- Destroyed → no damage
- Bitişik binaların birleşmesi
- Kaçırılmış küçük binalar
- Hizalama deneyi yapıldıysa görüntü kaymasıyla değişen tahmin
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

Sürüm numaraları ilk ortam smoke-test’i geçtikten sonra sabitlenir. PyTorch CUDA ve NVIDIA sürücü uyumu ilk fazda doğrulanır. ONNX Runtime sürüm uyumu, ek çalışma yapılırsa o aşamada değerlendirilir.

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
│   ├── damage_temporal.yaml  # İsteğe bağlı E1
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
python -m aeris benchmark --device cuda
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
16. IoU = 0,5 sınırı, eşit IoU sıralaması ve bire bir eşleştirme
17. İki bina birleşmesi, alan kapsamı = 0,5 sınırı ve ağır hasar tablosunda tek sayım
18. xView2 scorer biçimi ve beklenen skor, sınıflandırılmamış geometrinin işlenmesi
19. Oracle/aday eşiği seçimi, eşit skorlar ve eşleşmesiz kabul edilmiş adayların risk hesabı

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

Dondurulmuş sürümün PyTorch ile CPU ve GPU ölçümü M1 teslimidir. ONNX export ve parity kontrolü, yalnızca 11 Aralık tamponu boşsa denenir. C++ ONNX Runtime aracı sonraki sürüm için değerlendirilir.

## 13.2. Benchmark protokolü

- Aynı model, giriş boyutu ve önceden kaydedilen görüntü çiftleri, CPU ve GPU’da batch 1 ile çalıştırılır. Cihaz ve yazılım sürümleri ile bina aday sayıları kaydedilir.
- Model yükleme ve diskten görüntü okuma tamamlandıktan sonra, aynı girişler üzerinde 20–50 ısınma ve en az 200 uçtan uca ölçüm alınır. Uçtan uca süre ön işleme, bina lokalizasyonu, aday çıkarma, hasar sınıflandırması ve sonuç hazırlamayı kapsar.
- CUDA ölçümünde zamanlama sınırlarında gerekli senkronizasyon uygulanır. GPU tepe VRAM değeri ve ölçme yöntemi kaydedilir.
- CPU ve GPU için p50/p95 gecikme raporlanır. İmkân varsa model-only süre, CPU tepe RAM ve model boyutu eklenir.
- ONNX uygulanırsa aynı girişlerde PyTorch ile çıktı farkı ve gecikme karşılaştırılır.

## 13.3. Runtime doğrulaması

Temiz kurulumda çalışan PyTorch CPU çıkarımı temel çalışma yoludur. CUDA yolu yerel GPU üzerinde doğrulanır. ONNX denemesi yapılırsa durumu kısa bir sonuç kaydıyla belgelenir.

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

Nihai test tablosu, risk–coverage grafiği ve hata örnekleri README ve değerlendirme raporunda yer alır. Ek demo paneli zaman kalırsa rapordan alınan etiketli benchmark sonuçlarını gösterir. Kullanıcının yüklediği etiketsiz çiftte tahminler ve inceleme durumu gösterilir.

## 14.5. Arayüz ilkeleri

- Varsayılan çalışma yerel olacak.
- Kullanıcı görüntüleri dış sunucuya gönderilmeyecek.
- Sonuçlar araştırma tahmini olarak etiketlenecek.
- Lisanslı xBD görselleri izinsiz public demo içine gömülmeyecek.
- Gerçek tahminler ile örnek/sentetik veriler açıkça ayrılacak.

---

# 15. Ayrıntılı geliştirme takvimi

## 15.1. Planlama varsayımı

**21 Kasım – 12 Aralık 2026**, 22 takvim günü ve yaklaşık 60–80 saat net çalışma. Tarihler teslim hedefleridir. xView2 erişimi ve indirme ilk oturumun önceliğidir. İndirme sürerken ortam kurulumu ilerleyebilir. Gerekirse GPU ortam testi 22 Kasım sabahına kayar.

## 15.2. Gün bazında çalışma ve teslimler

| Tarih | İş | Teslim veya kapı |
|---|---|---|
| 21 Kasım | xView2 erişimi, paket koşulları ve challenge train indirmesini başlatma. Yerel depo, Python ortamı, NVIDIA/PyTorch ve CPU smoke testi. | Veri erişim durumu ve ilk ortam kaydı |
| 22–23 Kasım | Görüntü ve JSON şemasını inceleme, eşleşme, etiket normalizasyonu, poligon rasterization, manifest ve parser testleri. Resmî scorer’ı 1024×1024 sentetik maskelerle doğrulama. | Doğrulanmış veri okuma ve skor hattı |
| 24 Kasım | Olay ve sınıf desteği, sahne/mekânsal grup, aynı türden OOD olay seçimi, sızıntı denetimi ve split hash. | G0 |
| 25 Kasım | U-Net, BCE + Dice, 20–50 yamada smoke ve tiny-overfit. Tepe VRAM, epoch süresi ve veri okuma hızı. Darboğaz ölçülürse yerel önbellek. | Çalışan eğitim döngüsü ve [donanım ölçümü](localization-smoke.md) |
| 26 Kasım | Gerçek eğitim alt kümesiyle lokalizasyon eğitimi. Validation maskeleri, eşik seçimi, F1/IoU ve checkpoint. | [İlk bina segmentasyon modeli](localization-baseline.md) |
| 27–28 Kasım | Ağırlıklı CE ile gerçek bina crop’larında post-only hasar eğitimi. Validation harmonik F1 ile checkpoint, confusion matrix ve ham skor kaydı. | G1–G2 |
| 29–30 Kasım | Tahmin edilen adaylardan uçtan uca çıkarım ve xView2 maskeleri. Bire bir IoU, birleşme ve ağır hasar kategorilerinin sentetik kontrolleri. | Çalışan uçtan uca değerlendirme |
| 1 Aralık | CLI, JSON çıktısı, CPU smoke çıkarımı ve README kurulum adımları. | Etiketsiz görüntü çiftiyle çalıştırılabilir hat |
| 2 Aralık | Tampon. | M0 öncesi kurtarma zamanı |
| 3 Aralık | Validation üzerinde gerçek ve tahmin edilen bina sınırlarının kıyası. | M0 · G3 |
| 4 Aralık | Seçilen modelle ID kalibrasyon logits ve adayları. `T`, `τ_oracle` ve `τ_candidate` belirleme, model ve değerlendirme kurallarını dondurma, nihai ID/OOD testini tek koşuda çalıştırma. | Nihai skorlar ve bina bazlı hata tablosu |
| 5 Aralık | ID/OOD ham ve kalibre ECE/NLL, sabit eşik noktalarıyla oracle ve aday risk–coverage grafikleri. Yeterli blok varsa bootstrap, örnek hata galerisi. | G4 |
| 6 Aralık | Dondurulmuş PyTorch modeliyle CPU/GPU p50/p95 ve GPU tepe VRAM. Zaman kalırsa blur, JPEG ve kaydırma duyarlılığı. | Kaynak ve gecikme raporu |
| 7–8 Aralık | Zaman kalırsa validation’da Siamese `post/post` ve `pre/post` kıyası. | İsteğe bağlı E1 |
| 9 Aralık | Dondurulmuş post-only model, aday eşiği, maske ve inceleme işaretiyle sade yerel demo. | Çalışan demo |
| 10 Aralık | Model/data card, sınırlamalar, README başında sonuç tablosu ve eşik grafiği. Kaynak bağlantılarını ve kurulum adımlarını kontrol etme. | G5 |
| 11 Aralık | Yayın öncesi tampon, işler tamamlandıysa isteğe bağlı ONNX eşdeğerlik denemesi. | Yayın güvence zamanı |
| 12 Aralık | Temiz kurulum ve CLI/demonun uçtan uca doğrulaması, sonuç yazımı ve sürüm yayını. | M1 ve kapsam kapanışı |

M0, validation sonuçlarıyla gösterilir. M1’in demo modeli ve eşikleri 4 Aralık nihai değerlendirmesinde kullanılan sürümdür. E1 ve E2 validation ile sınırlı, keşifsel ek deneylerdir.

---

# 16. Karar kapıları ve durdurma kuralları

| Kapı | Koşul | Koşul karşılanmazsa |
|---|---|---|
| G0 | Kullanım koşulları, gerçek parser, etiket desteği, scorer ve bölme manifesti doğrulandı | Veriyi ve protokolü doğrulamaya devam et. Bekleme süresinde sentetik testleri ilerlet. |
| G1 | Tiny-overfit ve metrik testleri geçiyor | Veri, etiket ve eğitim döngüsünü incele. |
| G2 | İki eğitim hattı 4 GB içinde kararlı | Crop, batch veya model boyutunu küçült ve yeniden ölç. |
| G3 | Etiketsiz uçtan uca CLI ve validation değerlendirmesi çalışıyor | Tamponu çekirdek hattın onarımına ayır. |
| G4 | Dondurulmuş ID/OOD testi, bina hataları, kalibrasyon ve iki eşik grafiği hazır | Bilimsel teslimi öne al ve ek deneyleri çıkar. |
| G5 | Sade demo, model/data card, README sonuçları ve test akışı hazır | Yayın tamponunu kurulum, belge ve demo doğrulamasına ayır. |

### Kapsam azaltma sırası

- **Veri erişimi gecikirse.** G0 için kaynak erişimini takip et. Bu sırada sentetik çift ve maskelerle parser, scorer ve eşleştirme testleri ilerleyebilir. Gerçek eğitim ve ID/OOD teslim koşulları, gerçek veri elde edilince doğrulanır.
- **Takvim sarkarsa.** 11 Aralık tamponu önce yayın doğrulamasına ayrılır. Ardından E1/E2 ve sonrasında blur/JPEG/kaydırma deneyleri kapsamdan çıkar. Ana eşik deneyi, bina hata tablosu, PyTorch ölçümü, sade demo ve README korunur.
- **Lokalizasyon sorun çıkarırsa.** Oracle hasar modelinin sonuçlarını kaydet ve M0 tamponunu uçtan uca hattı onarmak için kullan. M0 statüsü CLI ile gerçek veri tahmini ve validation kıyasıyla doğrulanır.
- **OOD sınıf desteği zayıfsa.** Olay ve sınıf başına sayıları, `N/A` değerlerini ve yorum kapsamını göster.

---

# 17. Risk kayıtları

| Risk | Etki | Erken uyarı | Önlem |
|---|---|---|---|
| Dataset erişimi/lisans | Kritik | G0 geçilemiyor | Erişim ve kullanım koşullarını ilk günlerde doğrula |
| Split leakage | Kritik | Aynı scene birden çok split'te | Grup bazlı test ve hash |
| GPU OOM | Yüksek | İlk eğitimde bellek taşması | Küçük model/crop/batch |
| Eğitim çok yavaş | Yüksek | İlk epoch ve veri okuma ölçümünde düşük throughput | Ölçülen darboğaza göre crop önbelleği, alt küme veya deney sayısı |
| Minor/major ayrımı kötü | Yüksek | Confusion matrix yoğun | Hata analizi, weighted loss |
| Localization birleşme hatası | Orta–yüksek | Bitişik binalar tek aday | Önceden tanımlanan IoU ve birleşme ölçümleri |
| OOD çöküşü | Yüksek | Yeni afette büyük F1 düşüşü | Sonucu gizleme, sınırlama olarak raporla |
| OOD güven eşiği aktarımı | Yüksek | Sabit eşikte kapsama veya hata oranı değişiyor | ECE/NLL ve sabit eşik sonuçlarını ayrı raporla |
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
- Birden fazla binanın tek adayda birleşmesi
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

- Dataset ve split raporu, eğitim koşusu ile checkpoint kayıtları
- ID/OOD sınıf desteği, lokalizasyon skoru, hasar confusion matrix ve gerçek/tahmin edilen bina sınırı karşılaştırması
- Bire bir bina eşleştirmesi, birleşik aday ve ağır hasar hata tablosu
- Ham ve kalibre ECE/NLL, sabit eşik noktalarıyla ana ve uçtan uca risk–coverage grafikleri
- PyTorch CPU/GPU p50/p95, GPU tepe VRAM ve örnek hata galerisi
- E1/E2, görüntü bozulması veya ONNX sonuçları, ilgili ek çalışma tamamlandığında

## 19.3. Belgeler

- README başında nihai karşılaştırma tablosu ve ana risk–coverage grafiği
- Architecture, data card, model card, evaluation protocol ve limitations
- Çalıştırma ve yeniden üretim adımları, release notes
- Yayıma hazırlanan dokümanlarda doğrulanmış kaynak bağlantıları. Araştırma 2’den aktarılan atıflar gerçek kaynak bağlantısıyla yazılır. Yerel çalışma oturumuna bağlı bağlantılar yayın öncesinde kontrol edilir. Haricî skorlar kendi veri ve metrik protokolüyle birlikte verilir.

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
| Deep learning | PyTorch eğitim, transfer learning ve ağırlıklı çapraz entropi |
| Data engineering | Manifest, parser, validation, split |
| Scientific evaluation | F1, bina eşleştirmesi, OOD, kalibrasyon ve sabit eşik deneyi |
| Software architecture | Modüler paket, CLI, test, CI |
| Runtime engineering | PyTorch CPU/GPU p50/p95 ve GPU tepe VRAM |
| Problem solving | Hata analizi ve karar kapıları |
| Responsible AI | Manual review, model card, sınırlamalar |

Bu eşleştirme teknik portföy stratejisidir; herhangi bir kurumun işe alım kararı veya değerlendirme ağırlığı hakkında garanti değildir.

---

# 22. Nihai karar kaydı

| Karar | Seçim |
|---|---|
| Veri | xBD challenge train. G0 sınıf desteğine göre Tier 3 değerlendirilebilir. |
| Lokalizasyon | Afet öncesi RGB, hafif U-Net / ResNet18. Checkpoint ölçütü validation ikili piksel F1. |
| Hasar | Afet sonrası RGB crop, ResNet18 ve `min(4, sqrt(n_max / n_c))` ile ağırlıklı CE. Checkpoint ölçütü oracle validation harmonik damage F1. |
| Bina değerlendirmesi | `IoU ≥ 0,5` açgözlü bire bir eşleştirme. Bir adayın iki veya daha fazla gerçek binada ayrı ayrı sağladığı `> %50` alan kapsamı birleşmedir. |
| Görülmemiş afet | Öncelikle eğitimde temsil edilen bir türün tamamı dışarıda tutulan olayı. Nihai ID/OOD testi 4 Aralık. |
| Ana araştırma | Oracle crop risk–coverage, ID kalibrasyonda yaklaşık %80 kapsama ile seçilen sabit `τ_oracle`. |
| Demo | Aynı sıcaklık katsayısı, tahmin edilen adaylarda seçilen ayrı sabit `τ_candidate` ve 4 Aralık’ta test edilen model. |
| E1 | Aynı Siamese ağda `post/post` ve `pre/post`. İsteğe bağlı, validation ile sınırlı keşifsel kıyas. |
| Runtime | PyTorch CPU/GPU ölçümü zorunlu. ONNX 11 Aralık tamponu boşsa. C++ aracı sonraki sürüm. |
| Takvim | M0 3 Aralık. Tamponlar 2 ve 11 Aralık. M1 ve kapsam kapanışı 12 Aralık. |

---

# 23. İlk uygulama oturumu

İlk oturum veri erişimi ve çalışma ortamı doğrulamasıyla ilerler.

1. [xView2’nin resmî veri yolunu](https://github.com/DIUx-xView/xView2_baseline#data-downloads) kullanarak erişimi, hesap gereksinimini ve challenge train paketinin kullanım koşullarını doğrula. Erişim sağlandığında indirmeyi başlat.
2. İndirme sürerken yerel depo ve izole Python ortamını hazırla. Linux, Python, NVIDIA, CUDA ve PyTorch sürümlerini kaydet. CPU ve GPU için basit PyTorch işlemiyle çalışma ortamını doğrula.
3. İlk gerçek pre/post çiftini ve iki anotasyonu aç. Olay, sahne ve konum alanlarını, etiketleri ve piksel geometrisini incele.
4. Sentetik parser ve scorer kontrolleriyle başla. Veri, bölme ve eğitim sözleşmelerini 24 Kasım G0 kapısında doğrula.

---

# 24. Kaynaklar ve araştırma temeli

Bu ürün planı iki önceki AERIS araştırma belgesinden türetilmiştir:

- [Araştırma 1/2 — Bilimsel ve Teknik Ön Araştırma](AERIS_VISION_Arastirma_1_Bilimsel_Teknik_Rapor_2026-11-20.md)
- [Araştırma 2/2 — Literatür, Mimari Kararlar ve Teknik Fizibilite](Derin%20Ara%C5%9Ft%C4%B1rma.md)

Başlıca birincil teknik referanslar:

- [xBD makalesi](https://arxiv.org/abs/1911.09296)
- [TorchGeo xBD](https://docs.torchgeo.org/en/stable/api/datasets/xbd.html)
- [xView2 resmî baseline](https://github.com/DIUx-xView/xView2_baseline)
- [xView2 scoring](https://github.com/DIUx-xView/xView2_scoring)
- [Microsoft Siamese BDA](https://github.com/microsoft/building-damage-assessment-cnn-siamese)
- [Guo ve ark. — On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
- [Ovadia ve ark. — Can You Trust Your Model’s Uncertainty?](https://arxiv.org/abs/1906.02530)
- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [ONNX Runtime](https://onnxruntime.ai/docs/)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)

Bağımlılıkların güncel sürümleri, veri indirme koşulları ve lisans ayrıntıları uygulamanın ilk günlerinde yeniden doğrulanacaktır.

---

# 25. Proje tamamlanma tanımı

Portföy sürümü tamamlandığında aşağıdaki kanıtlar hazırdır.

- Gerçek xBD verisinde eğitilen iki checkpoint, kayıtlı split hash’i ve izlenebilir seçim metrikleri.
- Etiketsiz RGB pre/post görüntü çiftinden çalışan CLI, JSON çıktısı ve sade yerel demo.
- Dondurulmuş nihai ID/OOD testi, piksel ve bina metrikleri, birleşik aday ve ağır hasar hata tablosu.
- ID/OOD ham ve kalibre edilmiş ECE/NLL raporu, oracle ve aday risk–coverage grafikleri ve sabit eşik noktaları.
- Dondurulmuş PyTorch sürümü için CPU/GPU p50/p95 gecikme ve GPU tepe VRAM ölçümü.
- Geçen temiz kurulum ve temel testler, README sonuç tablosu, model/data card, örnek hatalar ve sınırlamalar.
- Veri ve üçüncü taraf kod kaynakları ile uygun yayın atıfları.

**Belge sonu — P01 / D03 / v1.1**
