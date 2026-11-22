# AERIS VISION — Derin Araştırma 2/2: Literatür, Mimari Kararlar ve Teknik Fizibilite

## Araştırmanın ana hükmü

İlk araştırmadaki ana fikir **büyük ölçüde doğru**, fakat ikinci araştırmanın sonucu projeyi önemli ölçüde keskinleştiriyor: AERIS'i “en büyük modeli eğiten proje” yapmaya çalışmamalıyız. Onu **küçük donanımda çalışan, bilimsel olarak doğru değerlendirilen, bilinmeyen afetlerde nasıl bozulduğu ölçülen, belirsiz durumlarda tahmin yapmamayı bilen ve gerçek bir yazılım sistemi gibi paketlenmiş bir Building Damage Assessment sistemi** hâline getirmeliyiz.

xBD/xView2 bu amaç için hâlâ doğru ana veri kümesi. Güncel xBD tanımı 850.736 bina anotasyonu ve yaklaşık 45.362 km² görüntü alanı içeriyor; afet öncesi ve sonrası görüntülerle bina footprint'leri ve hasar sınıfları sağlıyor. citeturn475880academia8turn301233search2 Resmî xView2 değerlendirmesinde toplam skorun yalnızca %30'u bina lokalizasyonundan, %70'i ise dört hasar sınıfının F1 değerlerinin harmonik ortalamasından geliyor. Dolayısıyla “binaları güzel segmentledim” tek başına yeterli değil; özellikle **minor** ve **major damage** başarısı kritik. citeturn301233search1

Projenin doğru iddiası şu olmalı:

> Afet öncesi ve sonrası uydu görüntülerinden bina lokalizasyonu ve dört seviyeli hasar sınıflandırması yapan; olaylar arası genelleme, güven kalibrasyonu, manuel incelemeye yönlendirme, hata analizi ve inference performansını ölçen araştırma amaçlı karar-destek sistemi.

“Operasyonel olarak güvenli bina belirliyor”, “saha incelemesinin yerini alıyor” veya “yüksek doğrulukla gerçek afet kararı veriyor” gibi iddialar yapılmamalı. xBD'nin ortak hasar ölçeği uydu görüntüsüne göre görsel sınıflandırma amacıyla geliştirilmiş; özellikle minor/major ayrımı, üstten çekilmiş optik görüntünün bazı yapısal hasarları göstermemesi nedeniyle intrinsik olarak belirsiz olabiliyor. ChangeMamba çalışması da minor ve major kategorilerinin semantik belirsizliğini ve ortorektifiye optik görüntülerin bazı yapısal hasarlarda yetersizliğini açıkça tartışıyor. citeturn475880search7turn167684search8

Bu yüzden AERIS'i “devasa” yapan şey model boyutu olmayacak. **Pipeline'ın kapanması** olacak:

**veri → leakage-safe split → localization → damage classification → OOD → calibration → abstention → robustness → ONNX → testler → demo → model card.**

Bu, 56 günlük takvimde üç ayrı proje hazırlayacak bir öğrenci için, ağır bir transformer veya foundation model yeniden üretiminden daha güçlü bir mühendislik sinyali.

## Veri, etiketler ve benchmark konusunda kesinleşenler

TorchGeo'nun güncel xBD entegrasyonu challenge training set için yaklaşık **7,8 GB**, test set için yaklaşık **2,6 GB** veri kullanıyor; görüntüler üç kanallı PNG ve maskeler background, no-damage, minor, major, destroyed biçiminde beş değerli sınıf yapısına dönüştürülüyor. TorchGeo ayrıca Tier3, holdout ve “full data” paketlerinin bu standart sınıfa dâhil olmadığını özellikle belirtiyor. citeturn301233search3turn475880search4 Resmî xView2 baseline deposu challenge'ın kullandığı veri için yaklaşık 10 GB sıkıştırılmış, 11 GB açılmış disk alanı bildiriyor; pratik boyut, seçilen alt paketlere göre değişebiliyor. citeturn301233search0

xBD hasar sınıflarının anlamı kabaca şöyle: **no damage**, görünür su/yapısal/yanık hasarı olmayan bina; **minor**, kısmi yanık, su/lava çevresi, eksik çatı elemanı veya görünür çatlak; **major**, kısmi duvar/çatı çökmesi veya yoğun su/çamur etkisi; **destroyed**, tamamen çökmüş, ciddi yanmış, su/çamurla örtülmüş ya da artık mevcut olmayan bina. citeturn475880search7turn475880search1 Bunlar yapı mühendisliği saha güvenlik sınıfları değil; bunu model kartında özellikle yazacağız.

Birinci araştırmada doğru tespit edilmiş ama ikinci araştırmada daha da önemli hâle gelen nokta **registrasyon**. Pre ve post görüntüler arasında piksel seviyesinde mükemmel hizalama varsayımı güvenli değil. Microsoft'un açık Siamese pipeline'ı bitemporal özellik farklarını kullanıyor; 2026'daki çalışmalar bile hizalama modüllerini hâlâ performansın ve genellemenin önemli bileşenlerinden biri olarak ele alıyor. citeturn686414view1turn267281search6 Bu nedenle projeye “±2 / ±4 / ±8 piksel yapay kaydırmada model ne kadar bozuluyor?” deneyi eklemek, sıradan bir model eğitiminden çok daha değerli olacak.

En kritik metodoloji kararı şu:

**İki ayrı değerlendirme protokolümüz olacak.**

Birincisi **benchmark protokolü**: xView2 scorer'a mümkün olduğunca sadık evaluation. İkincisi ise **OOD protokolü**: en az bir veya tercihen iki afet olayının tamamen eğitim dışında tutulduğu held-out-event değerlendirmesi.

TorchGeo'nun `xBDDistShift` sınıfı afet bazlı distribution shift fikrini doğrudan destekliyor; ancak hasar sınıflarını binary building maskesine dönüştürdüğü için dört sınıflı damage OOD testi olarak doğrudan kullanılamaz. citeturn301233search3 Biz kendi dört sınıflı event split'imizi üretmeliyiz.

Bu, literatürün 2026'da özellikle işaret ettiği probleme de birebir denk geliyor. Coğrafi bias çalışması, aynı veri dağılımında iyi çalışan BDA modellerinin farklı coğrafya, bina dokusu ve afetlerde ciddi performans kaybı yaşayabildiğini; özellikle intermediate hasar sınıflarının zorlandığını gösteriyor. citeturn267281search0

Alternatif veri kümeleri açısından karar net:

| Veri kümesi | Değeri | AERIS kararı |
|---|---|---|
| xBD/xView2 | Dört sınıflı bitemporal bina hasarı | **Ana veri** |
| SpaceNet | Çok büyük bina footprint kaynağı | Sadece opsiyonel localization pretraining |
| Sen1Floods11 | Sentinel-1 flood segmentation | Farklı problem; MVP dışı |
| BRIGHT | Optik + SAR, modern multimodal BDA | External validation / stretch |

SpaceNet 67.000 km²'den fazla görüntü ve 11 milyondan fazla bina footprint'i içeren geniş bir mapping ekosistemi sunuyor; fakat hasar severity etiketi sağlamıyor. citeturn855262search0 Sen1Floods11 11 taşkın olayını ve yaklaşık 14 GB'lık Sentinel-1 flood segmentation verisini kapsıyor, fakat AERIS'in dört sınıflı bina hasar problemiyle aynı görev değil. citeturn855262search2turn855262search3 BRIGHT ise 2025'te yayımlanan, 14 bölgeyi kapsayan, optik ve SAR kullanan güncel bir Building Damage Assessment benchmark'ı; çok değerli ama ilk teslim için scope'u gereksiz büyütüyor. citeturn855262search7turn855262search6

Lisans konusunda da ilk araştırmayı biraz sıkılaştırıyorum. Maxar Open Data Programı AWS'de **CC BY-NC 4.0** olarak listeleniyor. citeturn301233search4 xBD aynalarında CC BY-NC-SA ifadeleri de görülüyor; dolayısıyla ham imagery'yi GitHub'a koymayacağız. Güvenli politika: **kod + downloader talimatı + manifest generator + sentetik test örneği**, fakat veri yok. Nihai README yayınlanmadan önce xView2 hesabındaki güncel dataset terms ayrıca manuel kontrol edilmeli.

## Güncel literatürün bize söylediği şey

2024–2026 literatüründe en önemli bulgu şu: araştırma artık yalnızca “daha iyi segmentation architecture” peşinde değil. Domain shift, class imbalance, multimodal veri, uncertainty ve instance-level değerlendirme öne çıkmış durumda.

**BAT — Bitemporal Attention Transformer (2024)**, EfficientNetV2S encoder ile Swin tabanlı attention/decoder yaklaşımı kullanıyor ve hasar sınıflarının ordinal yapısını loss'a dâhil ediyor. Fakat yazarların eğitim ortamı RTX 3090, batch 7 ve 768×768 crop. citeturn167684search3 Bu yüzden BAT'ı RTX 3050 Ti 4 GB üzerinde yeniden üretmeye çalışmak kötü zaman kullanımı. Ama **ordinal loss** fikri stretch deney olarak değerli.

**ChangeMamba / MambaBDA (2024)**, günümüzde güçlü açık kaynak referanslardan biri. Güncel resmî repo MambaBDA için Linux kodunu, pretrained VMamba encoder'ları ve training pipeline'ını sağlıyor; Small modeli için örnek komut batch 16, crop 512 ve 80.000 iterasyon kullanıyor. citeturn686414view0 Repo ayrıca 2026'da reorganize edildiğini ve yeni kodla üretilen skorların yayınlanan makaledeki skorlarla birebir uyuşmayabileceğini açıkça belirtiyor. citeturn686414view0 Bu bizim için çok öğretici: modern görünmek uğruna ChangeMamba full training'e girmek yerine, **checkpoint/inference reference veya küçük fine-tune stretch** olarak tutacağız.

**DeepDamageNet (2024)** iki aşamalı building segmentation + damage classification pipeline'ı kullanıyor ve combined F1=0,66 raporluyor. Aynı çalışma, farklı afetlerde damage classification'ın localization'dan belirgin biçimde daha zor olduğunu vurguluyor. citeturn167684academia7 Bu, ilk araştırmadaki iki aşamalı mimari kararını destekliyor.

**Single-Temporal Change Adaptation (2024)**, yeni hedef bölgeye ait yalnız pre-disaster imagery kullanarak domain adaptation araştırıyor. citeturn267281search5 Biz bunu implement etmeyeceğiz, fakat önemli ders şu: “yeni afet performansı” mimari seçiminden bağımsız ciddi bir problem.

**ViPDE (2025)** SAM bilgisi ve visual prompting kullanarak foundation-model tabanlı BDA araştırıyor. citeturn267281search2 Bilimsel olarak güncel ama bizim MVP için yanlış seçim. 4 GB VRAM, 56 gün ve üç proje varken foundation model eklemek teknik sinyalden çok kapsam riski yaratıyor.

**Geographic Bias Analysis (2026)**, AERIS'in en önemli bilimsel yönünü doğrulayan çalışma. Modellerin yeni bölgelere ve yapılara genellenmesindeki zorlukları doğrudan inceliyor; özellikle minor ve major damage tarafında domain shift sorununu gösteriyor. citeturn267281search0

**Cost-Sensitive Cross-Disaster BDA (2026)**, sınıf dengesizliği ve afetler arası aktarım üzerinde çalışıyor. citeturn267281search1 Bundan çıkaracağımız şey 15 farklı loss denemek değil; **CE vs weighted CE/focal** gibi tek temiz ablation yapmak.

**Improved MambdaBDA (2026)** Focal Loss, attention gates ve alignment modülünü cross-domain koşullarda araştırıyor. citeturn267281search6 Biz alignment module yazmadan önce registrasyon stres testi yapacağız. Eğer küçük pixel shift modelimizi gerçekten kırıyorsa, ancak o zaman alignment özelliği geliştirmeye değecek.

**Insta-BDA (2026)** pixel-level maskelerden instance-level building damage/counting'e geçmenin değerini gösteriyor ve cross-paper skor karşılaştırmalarının protokol farkları nedeniyle dikkatle yorumlanması gerektiğini de açıkça belirtiyor. citeturn267281search8 AERIS demosunun bina bazlı tablo/özet vermesi bu nedenle çok mantıklı.

## Mimari kararı: üç rota, biri ana rota

### Ana rota — hafif iki aşamalı sistem

Bu en düşük riskli ve en öğretici rota.

**Localization modeli:** pre-disaster RGB → bina maskesi. Küçük encoder'lı U-Net ailesi; örneğin ResNet18, MobileNetV3 veya EfficientNet-B0 seviyesinde pretrained backbone.

Başlangıç varsayımı:

- 256×256 crop,
- batch 2–4,
- AMP,
- BCE + Dice,
- ImageNet pretrained encoder.

Bu batch/crop değerleri **ölçülmüş garanti değil**, 4 GB VRAM için güvenli smoke-test başlangıç noktası. PyTorch AMP, FP16/BF16 gibi daha düşük hassasiyet türlerini kontrollü biçimde kullanarak bellek ve throughput avantajı sağlayabiliyor. citeturn262788search1turn262788search2

**Damage classifier:** önce ground-truth building polygon/crop kullanacağız. Böylece localization kötü olsa bile damage classifier'ın kendisinin ne kadar iyi olduğunu görebiliriz. Buna **oracle-footprint baseline** diyeceğiz.

Sonra predicted maskelerden bina crop'ları çıkarıp aynı classifier'a vereceğiz.

Bu bize çok güçlü bir mühendislik grafiği üretir:

**GT footprint damage F1 → predicted footprint damage F1**

İki skor arasındaki fark, localization hatasının damage pipeline'a ne kadar yayıldığını gösterir.

### İkinci rota — lightweight Siamese

Aynı pretrained küçük encoder'ın pre ve post görüntüleri işlemesi; feature concat/difference; küçük decoder veya damage head.

Amaç:

**“Pre/post temporal reasoning gerçekten post-only modelden daha iyi mi?”**

Microsoft'un açık sistemi shared Siamese segmentation encoder'ları ve feature difference üzerinden damage classification yapıyor; bu yaklaşımın açık kodu ve Docker pipeline'ı mevcut. citeturn258483search2turn686414view1

Bunu “dev model” olarak değil, bilimsel karşılaştırma olarak kullanacağız.

### Üçüncü rota — Mamba/Transformer stretch

ChangeMamba'yı ortamda ayağa kaldırmak, checkpoint inference yapmak ve donanım uygunsa küçük crop'la kısa fine-tuning denemek güzel bir stretch olur. Ama ana teslim **değil**. Resmî eğitim ayarları bizim VRAM bütçemizden ciddi biçimde büyük. citeturn686414view0turn167684search3

Bu nedenle mimari öncelik:

**A > B >>> C**

Önerdiğim deney çekirdeği:

| Deney | Soru |
|---|---|
| E0 | 200–500 örneği model overfit edebiliyor mu? |
| E1 | Post-only vs pre+post |
| E2 | CE vs weighted/focal |
| E3 | Random split vs held-out-event |
| E4 | GT footprint vs predicted footprint |
| E5 | Raw confidence vs temperature-scaled |
| E6 | Abstention ile risk düşüyor mu? |
| E7 | Blur / JPEG / downsample robustness |
| E8 | ±2 / ±4 / ±8 px misregistration |
| E9 | Small vs medium encoder: F1 / latency / VRAM |
| E10 | PyTorch vs ONNX Runtime |

Bu matris bana göre AERIS'in en etkileyici kısmı olacak. Çünkü başvurudaki mühendis şunu görecek:

**“Model kurmuş” değil, “deney tasarlamış.”**

## Güvenilirlik ve ürün mühendisliği

Confidence skorunu doğrudan “olasılık” kabul etmeyeceğiz. Modern neural network'lerin softmax skorlarının kötü kalibre olabildiği; **temperature scaling** gibi basit post-hoc yöntemlerin validation set üzerinde güven kalibrasyonu için etkili olabildiği klasik çalışmalarda gösteriliyor. citeturn270161search1

AERIS'te:

1. Validation logits alınacak.
2. Temperature scaling sadece validation'da fit edilecek.
3. Testte reliability diagram ve ECE raporlanacak.
4. `max_probability < τ` ise çıktı **manual review** olacak.

Selective classification literatürü, modelin riskli örneklerde tahminden kaçınmasını doğrudan risk–coverage çerçevesinde tanımlıyor. citeturn270161search6turn270161search9 Bu nedenle tek bir confidence threshold göstermek yerine **risk–coverage curve** raporlayacağız.

Örneğin:

**%100 coverage → %70 accuracy**  
**%80 coverage → %78 accuracy**  
**%60 coverage → %85 accuracy**

Bunlar örnek değerler; gerçek sonuçlar değil. Ama raporlamanın mantığı bu.

Robustness testleri de kontrollü olacak:

- Gaussian blur,
- downsampling,
- JPEG compression,
- ±2/4/8 px misregistration.

Her testte:

`baseline F1 → perturbed F1 → delta`

gösterilecek.

Kod tarafında repo, training script karmaşası değil, düzgün bir sistem gibi olmalı:

```text
configs/
src/aeris/data/
src/aeris/models/
src/aeris/train/
src/aeris/eval/
src/aeris/infer/
src/aeris/demo/
tests/
reports/
```

Her deneyde git commit, config, seed, dataset manifest hash, environment/package sürümleri, checkpoint hash ve metrics JSON kaydedilmeli. PyTorch deterministic algoritmaları bazı işlemlerde performans bedeli getirebildiğinden, hedef “sihirli şekilde %100 deterministik” demek değil; kullanılan determinism ayarlarını açıkça belgelemek olmalı. citeturn262788search10turn262788search14

ONNX en sonda gelecek. ONNX Runtime hem NVIDIA için CUDA Execution Provider hem de CPU fallback sunuyor. citeturn262788search0turn262788search4 Benchmark:

- 20–50 warm-up,
- ≥200 inference,
- batch 1,
- p50 / p95 latency,
- peak RAM / VRAM,
- PyTorch–ONNX output difference.

TensorRT şart değil.

CI tarafında GitHub Actions + pytest yeterli. GitHub'ın resmi Python Actions kılavuzu `setup-python`, pytest ve test-result artifact akışını destekliyor. citeturn262788search11 Dataset ve GPU gerektirmeyen testler CI'da çalışacak:

- label parser,
- split leakage guard,
- scorer,
- crop geometry,
- calibration serialization,
- ONNX CPU smoke test.

Demo ise yalnız bir **inceleme arayüzü** olacak. Pre/post görüntü, bina overlay, dört sınıf olasılığı, manual-review durumu, latency ve model-card linki. Arayüzü güzel yaparız ama CV'de vurgu “Streamlit yaptım” olmayacak.

## BAYKAR ve HAVELSAN açısından neden güçlü?

BAYKAR'ın **2027 Yaz Dönemi Yapay Zekâ Yazılım Geliştirme** ilanı doğrudan şu alanları sayıyor: makine öğrenmesi, derin öğrenme, CNN/görüntü tabanlı yapay zekâ, bilgisayarlı görü, veri yönetim algoritmaları, büyük veri süreçleri ve yapay zekâ sistemlerinin çalışacağı yazılım/işletim sistemi mimarileri. citeturn703593search1

AERIS bunların birkaçını aynı projede ispatlayabilir:

| BAYKAR ilanındaki alan | AERIS kanıtı |
|---|---|
| Computer vision | bina segmentation |
| CNN / deep learning | localization + damage model |
| Machine learning | training, loss, calibration |
| Data management | manifest, split, data pipeline |
| AI software architecture | modular repo + configs |
| Model runtime | ONNX CPU/GPU benchmark |
| Model validation | OOD, robustness, failure analysis |

Bu, “BAYKAR kesin bundan etkilenir” demek değildir; işe alım tercihlerini uyduramayız. Fakat **ilanla teknik eşleşme doğrudan doğrulanabiliyor.** citeturn703593search1

BAYKAR 2027 yaz stajı başvuru son tarihi resmî sayfada **15 Ocak 2027** olarak yayımlanmış durumda. citeturn703593search0

HAVELSAN açısından AERIS, yapay zekâ ve analiz tarafında güçlü bir teknik portföy kanıtı olur; ancak HİT'in resmî sayfası seçim sürecinde GPA, genel yetenek ve İngilizce sınavı bulunduğunu belirtiyor. Ayrıca 3. veya 4. sınıfta olma ve ≥3.00 GPA şartı yayımlanmış durumda. citeturn703593search8 Dolayısıyla proje HAVELSAN için de değerli, fakat uygunluk ve resmî seçim sürecini ikame etmiyor.

## İlk araştırma için hüküm ve ürün planına taşınacak kararlar

İlk araştırmadaki önerileri ikinci araştırma sonucunda şöyle sınıflandırıyorum:

| İlk araştırma önerisi | Hüküm |
|---|---|
| xBD/xView2 ana veri | **DOĞRULANDI** |
| Hafif iki aşamalı ana sistem | **DOĞRULANDI, fakat daha da sadeleştirilmeli** |
| Siamese ikinci model | **DOĞRULANDI, lightweight tutulmalı** |
| Event-based split | **DOĞRULANDI ve projenin ana araştırma ekseni olmalı** |
| Resmî xView2 score | **DOĞRULANDI** |
| Robustness testleri | **DOĞRULANDI** |
| ONNX | **DOĞRULANDI, fakat projenin son aşaması** |
| Foundation model / SAM | **MVP için reddedildi** |
| BRIGHT external validation | **Stretch / zaman kalırsa** |

Üçüncü adımda hazırlanacak ürün planına geçecek **MVP** şu:

**xBD manifest + leakage-safe split → hafif localization → oracle-footprint pre/post damage classifier → predicted-footprint end-to-end → xView2 scorer → held-out-event OOD → CE vs weighted/focal → temperature scaling → abstention → robustness → ONNX → model card → failure gallery → demo.**

Stretch sırası:

**lightweight Siamese → ordinal-aware loss → registration/alignment deneyi → ChangeMamba inference → BRIGHT external validation.**

Şimdiden kesinlikle yapmayacaklarımız:

**full BAT reproduction, ağır 768–1024 transformer training, SAM/foundation-model fine-tuning, SAR fusion, TensorRT/custom CUDA kernel, gerçek zamanlı harita platformu ve “operasyonel bina güvenliği” iddiası.**

Karar kapıları da ürün planında aynen kullanılmalı:

**G0 — Veri:** dataset + lisans + manifest doğrulandı mı?  
**G1 — Pipeline:** tiny subset overfit + scorer unit test geçiyor mu?  
**G2 — Donanım:** 4 GB içinde stabil mi ve gerçek throughput ölçüldü mü?  
**G3 — Bilim:** baseline + event-OOD + confusion matrix tamam mı?  
**G4 — Güvenilirlik:** calibration + robustness var mı?  
**G5 — Deployment:** ancak bundan sonra ONNX.  
**G6 — Stretch:** MVP tamamlanmadan Mamba/BRIGHT/SAM yok.

Bence ikinci araştırmanın birinci araştırmaya kattığı en önemli şey şu:

**AERIS artık “uydu görüntülerinden hasar bulan bir AI projesi” değil.**

**AERIS = küçük donanımda çalışan, domain shift'i ölçen, güvenini kalibre eden, gerektiğinde “bilmiyorum” diyebilen ve reproducible biçimde deploy edilen bir afet görüntü analizi araştırma sistemi.**

Bu profil, modelin kaç milyon parametre olduğundan çok daha güçlü bir mühendislik hikâyesi oluşturuyor.

Raporun düzenlenmiş tam sürümleri:

[DOCX — AERIS VISION Derin Araştırma 2/2](sandbox:/mnt/data/AERIS_VISION_Derin_Arastirma_2_2_2026-11-20.docx)

[Markdown — AERIS VISION Derin Araştırma 2/2](sandbox:/mnt/data/AERIS_VISION_Derin_Arastirma_2_2_2026-11-20.md)