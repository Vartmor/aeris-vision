---
title: "AERIS VISION — Bilimsel ve Teknik Ön Araştırma"
subtitle: "2027 Savunma Teknolojileri Staj Portföyü | Proje 01 / Araştırma 01"
author: "Bağımsız teknik araştırma"
date: "20 Kasım 2026"
lang: tr-TR
---

# AERIS VISION

## Uydu Görüntülerinden Afet Sonrası Bina Hasarı Analizi

**Bilimsel ve Teknik Ön Araştırma · v1.0 · 20 Kasım 2026**  
**Doküman kimliği:** P01–R01 / planlanan 9 belgeli araştırma-geliştirme setinin 1. belgesi  
**Hazırlık amacı:** 2027 yazı BAYKAR Yapay Zekâ Yazılım Geliştirme / HAVELSAN staj hedefleri için özgün ve ölçülebilir bir mühendislik portföyü.

> **Dokümanın statüsü:** Bu bir *araştırma ve karar belgesidir.* Henüz yazılmış AERIS kodu, eğitilmiş model, koşulmuş deney veya elde edilmiş performans sonucu bulunmamaktadır. Buradaki mimari, test ve teknoloji tercihleri araştırmadan türetilen önerilerdir. İkinci araştırma bunları karşılaştıracak; üçüncü belge kesin ürün/geliştirme planına dönüştürecektir.

## Yönetici özeti ve araştırma kararı

**Araştırma kararı:** AERIS'i yalnızca “görüntü üzerinde hasarlı bina boyayan” bir demo olarak değil, **afetler arası genelleme, belirsizlik ve kısıtlı hesaplama kaynaklarında çıkarım** üzerine odaklanan bir uzaktan algılama / güvenilir yapay zekâ araştırma prototipi olarak konumlandırmak mantıklı.

Önerilen çekirdek problem: Afet öncesi ve sonrası yüksek çözünürlüklü RGB uydu görüntülerinden bina konumlarını ve görünür hasar seviyelerini tahmin etmek; başarımı sınıf bazında ve daha önce görülmemiş afet olaylarında ölçmek; belirsiz vakaları otomatik kesin yargıya dönüştürmemek.

Bunun gerekçesi dört kanıta dayanıyor:

1. **İş ilanı eşleşmesi:** BAYKAR'ın 2027 “Yapay Zekâ Yazılım Geliştirme” ilanında CNN, görüntü tabanlı yapay zekâ, derin öğrenme, veri yönetimi ve AI çalışma ortamları doğrudan sayılıyor. AERIS bu yeteneklerin teknik kanıtını üretebilir [S1].
2. **Gerçek benchmark:** xBD veri kümesi 19 doğal afet olayı, 22.068 görüntü ve 850.736 bina poligonu içeriyor. xView2'nin kamuya açık puanlama kodu mevcut [S2–S4].
3. **Araştırma boşluğu:** 2025–2026 literatürü sınıf dengesizliği, farklı coğrafyalara/afetlere geçişte performans kaybı ve sınırlı etiketli veri problemlerinin hâlen önemli olduğunu gösteriyor [S8–S11].
4. **Donanım ve süre:** 4 GB VRAM'li kişisel GPU ile devasa temel modelleri sıfırdan eğitmek uygun değil. Hafif iki aşamalı mimari, aktarım öğrenmesi ve kontrollü deneyler daha savunulabilir bir yaklaşım.

**Ön karar:** En iyi araştırma anlatısı, “en yüksek benchmark skorunu aldım” değil; **“aynı problemin doğruluk–genelleme–güvenilirlik–gecikme dengelerini sistematik biçimde ölçtüm”** olmalıdır.

**Öncelik şeması:** (A) doğrulanmış veri ve değerlendirme hattı → (B) temel model → (C) zaman öncesi/sonrası karşılaştırması → (D) olaylar arası test → (E) belirsizlik ve hız ölçümü → (F) şeffaf demo. Bu sıra üçüncü belgede takvime bağlanacaktır.

---

# 1. Araştırma sorusu, problem sınırı ve kullanım senaryosu

## 1.1. Çözülecek problem

Bir doğal afetin ardından çekilen uydu görüntüleri, geniş bir alanı insanı riske atmadan görme avantajı sunar. Ancak görüntüleri tek tek inceleyip binaları işaretlemek, hasar seviyesini tahmin etmek ve sonucu haritaya aktarmak zaman alır. xView2 girişimi tam bu insani yardım/afet müdahalesi problemine odaklanmıştır [S2].

AERIS'in araştırma sorusu:

> **Öncesi/sonrası optik uydu görüntüleri kullanarak bina hasarı tahminini ne kadar güvenilir, afetler arasında ne kadar genellenebilir ve düşük donanım maliyetiyle ne kadar hızlı yapabiliriz?**

Sistem iki farklı teknik görevi birbirine bağlar:

- **Bina yerelleştirme (localization / segmentation):** Hangi pikseller binaya ait? Bina sınırları ve örnekleri nasıl bulunur?
- **Hasar derecelendirme (damage classification):** Bina, görüntüdeki görünür işaretlere göre hangi hasar kategorisinde?

Dört hasar kategorisi **no damage, minor damage, major damage, destroyed** olarak tanımlanır. Piksel sınıflandırmasında ayrıca **background** vardır. Bu beş değer ile dört hasar sınıfı birbirine karıştırılmamalıdır [S3].

## 1.2. Hedef kullanıcı ve araştırma kullanım senaryosu

Hedef kullanıcı, afet sonrası uzaktan algılama verilerini inceleyen bir analist veya araştırmacıdır. Örnek iş akışı: iki eşlenik görüntü yüklenir; bina maskeleri ve görünür hasar etiketleri üretilir; düşük güvenli/şüpheli bölgeler öne çıkar; kullanıcı hasar dağılımını, örnek hataları ve model sınırlarını görür. Sistem, **insanın doğrulamasına yardımcı olan araştırma ekranı** niteliğindedir.

**Bilinçli olarak kapsam dışı:** Can güvenliği bakımından kesin bina kullanılabilirliği kararı, yapısal mühendislik sertifikası, sahada doğrulanmış arama-kurtarma önceliklendirmesi, gerçek zamanlı uydu veri tedariki, silah/hedef tespit sistemi ve operasyonel güvenlik garantisi. xBD'nin kendi hasar ölçeği de yerinde yapılan uzman hasar tespitinin yerine geçecek otoritatif bir değerlendirme olarak tasarlanmamıştır [S4].

## 1.3. Temel belirsizlik

Optik uydu görüntüsü bir yapının içini veya görünmeyen cephesini değerlendiremez. Çatı görüntüsü, gölge, bulut, duman, değişen bakış açısı, zamansal fark ve çözünürlük sınırlamaları, “görüntüde hasarlı görünmüyor” ifadesini “yapısal olarak sağlam” anlamına dönüştürmez [S4]. Bu sınırlama ürün iletişiminde açıkça yazılmalıdır.

# 2. BAYKAR / HAVELSAN portföyüne katkı analizi

## 2.1. Doğrudan belgelenen ilan eşleşmesi

BAYKAR'ın 2027 yaz dönemi Yapay Zekâ Yazılım Geliştirme ilanı; makine ve derin öğrenme, CNN/görüntü sistemleri, veri yönetimi, sinyal işleme ve AI çalışma ortamları gibi konuları listeliyor. “Bu proje yapılırsa işe alınır” yönünde kanıt yoktur; ama aşağıdaki portföy eşleştirmesi ilandaki teknik başlıklardan **mantıksal çıkarımdır** [S1].

| İlanla eşleşen beceri | AERIS'te doğrulanabilir kanıt | Sunulacak materyal |
|---|---|---|
| CNN ve görüntü tabanlı AI | Hafif segmentasyon + hasar sınıflandırma | Mimari diyagramı, model kodu |
| Derin öğrenme | Öğrenme eğrileri, kayıp/optimizasyon analizi | Deney raporu, checkpoint notları |
| Veri yönetimi | Veri manifesti, sürümleme, denetim ve olay-bazlı bölme | Veri kartı, split manifesti |
| Yazılım mimarisi | Ayrık eğitim, değerlendirme ve çıkarım paketleri | Testler, CLI, Dockerfile |
| AI çalışma ortamları | Yerel GPU, CPU ve ONNX kıyaslaması | Gecikme/bellek raporu |
| Yöntemsel eleştiri | Farklı afetlerde bozulma ve hata vakaları | OOD raporu, failure gallery |

HAVELSAN bağlantısı daha genel ve dolaylıdır: proje, yapay zekâ ve görüntü işleme mühendisliğine yönelik yetenek kanıtı sağlar; ancak **HAVELSAN HİT seçiminde projenin resmî bir değerlendirme ağırlığı olduğu iddia edilmemelidir**. Uygunluk, not ve sınav koşulları ayrı süreçtir.

## 2.2. Değer önerisi nasıl ifade edilmeli?

Zayıf portföy ifadesi: “Python ve PyTorch ile afet tespiti yaptım.”

Güçlü fakat yalnızca deneylerden sonra kullanılabilecek ifade: “Afet öncesi/sonrası görüntüler üzerinde yeniden üretilebilir bir bina hasarı değerlendirme hattı geliştirdim; veri sızıntısını önlemek için olay ve kaynak görüntü gruplarına göre test ettim; sınıf bazlı F1, görülememiş afet performansı ve çıkarım gecikmesini raporladım.”

Buradaki fark, **iddia değil kanıt** üretmektir. Portföyün değeri kod satırı sayısında değil, teknik kararların ölçülebilirliğindedir.

# 3. Veri ekosistemi: xBD, xView2, BRIGHT

## 3.1. Birincil veri: xBD / xView2

2019 tarihli xBD çalışması, afet öncesi/sonrası uydu görüntülerini bina poligonları ve hasar etiketleriyle birlikte tanımlar. Tam veri kümesi için makalede **22.068 görüntü, 19 afet, 850.736 bina poligonu ve yaklaşık 45.362 km²** kapsam verilir [S4]. Ön yayınların bazıları yaklaşık 700 bin poligon gibi erken dönem rakamları verir; bu raporda nihai xBD makalesindeki değerler esas alınmıştır.

TorchGeo'nun kararlı belgelerinde `xBD` yükleyicisi, challenge **training ~7,8 GB** ve **test ~2,6 GB** paketlerini kullandığını belirtir. `holdout`, `tier3` ve diğer paketler aynı yükleyicinin otomatik kapsamı değildir. Görüntüler RGB PNG, maske değerleri 0–4 sınıfıdır [S3].

**Önerilen öğrenme ve deney sırası:**

1. Küçük sayıda görüntüyle dosya/etiket şemasını doğrulama.
2. Veri kümesinden olay adı, kaynak kare, pre/post eşleşmesi ve hasar dağılımını içeren bir manifest çıkarma.
3. Veri kümesindeki eğitim bölümünden ayrı bir geliştirme/test planı kurma.
4. Önce **gerçek bina poligonundan kesilmiş patch** ile hasar sınıflandırma sınırı; sonra **tahmin edilen bina maskesi** ile uçtan uca sistemi değerlendirme.

Son madde kritiktir: Gerçek etiketlerle mükemmel bina konumu verilerek test edilen sınıflandırıcı ile modelin kendi tespit ettiği binalarla çalışan uçtan uca sistem **aynı görev ve başarı skoru değildir**.

## 3.2. Sınıflar ve dengesizlik

| Piksel maskesi | Etiket | Yorum |
|---|---|---|
| 0 | Background | Bina dışı pikseller |
| 1 | No damage | Görüntüde belirgin hasar işareti yok |
| 2 | Minor damage | Sınırlı görünür hasar |
| 3 | Major damage | Önemli görünür hasar |
| 4 | Destroyed | Belirgin yıkım |

**Önemli:** Bu etiketler xBD'nin görünür uydu bulgularına dayalı sınıflarıdır; yapı mühendisliği raporu değildir. Afetler arasında dağılım ve görsel desenler farklılık gösterir. Tek bir aggregate accuracy raporlamak özellikle nadir sınıfları görünmez kılabilir [S4, S10].

## 3.3. Görüntü geometrisi ve etiket kalitesi

xBD makalesi bakış açısı, çözünürlük (GSD), coğrafi kapsam ve sınıflandırma gürültüsünü kaynak sınırlamaları olarak tartışır. Önce ve sonra görüntülerinin birebir eşlenik olduğu varsayılmamalı; piksel düzeyinde kayma değişim algılamasını bozabilir [S4].

Veri denetiminde araştırılacak alanlar:

- `pre_image` ve `post_image` dosyalarının olay/kare ID eşleşmesi.
- Poligonların görüntü sınırını aşıp aşmadığı, kayıp/sıfır alanlı etiketler.
- Aynı yerin farklı tile ve augmentasyonlarının farklı kümelere düşüp düşmediği.
- Düşük çözünürlüklü bina örnekleri, çok küçük poligonlar, maskelerin sınır davranışı.
- Sınıf frekansları **görüntü**, **bina örneği**, **piksel** bazında ayrı sayılması.
- Görüntü başına bina yoğunluğu ve afet türüne göre dağılım.

## 3.4. Dış doğrulama adayı: BRIGHT (2025)

BRIGHT, afet öncesi optik ve sonrası SAR (sentetik açıklıklı radar) görüntüleriyle çok modlu bina hasarı çalışmaları için geliştirilmiş bir 2025 veri kümesidir. Çalışmada çok farklı afet ve bölgelere yayılan optik/SAR örnekleri tanıtılır; veri ve benchmark kodu erişim yolları yayımlanmıştır [S12].

**Amaç:** AERIS'in yalnızca RGB üzerinde çalıştığı ilk sürümünü BRIGHT ile doğrudan aynı görevmiş gibi puanlamak **metodolojik olarak yanlış** olabilir; sensör türü, etiketleme ve görüntü geometrisi farklıdır. BRIGHT, ikinci araştırmada çapraz-modal araştırma yönü olarak incelenmeli, ilk teslim için zorunlu tutulmamalıdır.

## 3.5. Veri hakları ve yayımlama

Maxar tabanlı optik görüntülerin lisanslarında araştırma / ticari olmayan kullanım kısıtları görülebiliyor. xBD için üçüncü taraf veri kataloğu ticari olmayan kullanım şartı belirtiyor; AWS Maxar Open Data kaydı da Creative Commons Attribution Non Commercial bilgisini veriyor [S13–S14]. **Ancak uygulamadan önce xView2 indirilen paketinin kendi kullanım koşulları ve her görüntünün kaynak lisansı doğrulanmalıdır.** Kod deposunun lisansı ile görüntü verisinin lisansı aynı şey değildir.

GitHub için güvenli varsayılan: **Ham xBD görüntülerini ve tam veri paketini depoya yükleme; indirilebilir veri kurulum talimatı, manifest üreticisi, küçük sentetik test verisi ve uygun atıflar yayımla.** Ekran görüntüsü / demo görsellerinin yeniden paylaşımına da kaynak lisansa göre karar ver.

# 4. Akademik ve açık kaynak teknoloji manzarası

## 4.1. İncelenen temel çalışmalar

| Kaynak | Yaklaşım ve katkı | AERIS için çıkarım | Sınırlama / dikkat |
|---|---|---|---|
| xBD (2019) [S4] | Veri kümesi, hasar ölçeği, olay çeşitliliği | Resmî veri ve problem tanımı | Hasar ölçeği saha denetimi değil |
| xView2 baseline [S5] | Bina yerelleştirme + hasar sınıflandırma | Tekrarlanabilir referans nokta | Eski bağımlılıklar olabilir |
| Microsoft Siamese CNN–U-Net [S6] | Pre/post paylaşımlı özellik çıkarma, iki aşamalı eğitim | Mimari karşılaştırma iskeleti | Oldukça ağır/eskimiş kurulum olabilir |
| Strong Baseline [S7] | PyTorch Lightning/Hydra ile güçlü temel | Deney tasarımı, ablation fikri | Cluster/HPC varsayımları var |
| DisasterAdaptiveNet (2025) [S8] | Çoklu tehlike türleri ve sağlamlık | Afet farklarının özellikle incelenmesi | Reprodüksiyon maliyeti araştırılmalı |
| Geographic Bias Analysis (2026) [S9] | Coğrafi OOD ve genelleme analizi | Ayrı olay/coğrafya testleri vazgeçilmez | Raporlanan skorlar kendi protokollerine ait |
| Cost-Sensitive Learning (2026) [S10] | Sınıf dengesizliği ve afetler arası geçiş | Weighted loss / maliyet duyarlı tasarım | Veri/etiket koşulları farklı olabilir |
| Insta-BDA (2026) [S11] | Nesne-bazlı sayım ve model füzyonu | Pixel-F1 tek başına yetmez | Foundation model maliyeti yüksek olabilir |
| BRIGHT (2025) [S12] | Optik + SAR ve çapraz-modal benchmark | Gelecek genişleme yönü | Ayrı sensör modalitesi / lisans |

Bu tablo **literatürde mevcut olduklarını ve tasarım fikirlerini** gösterir; skorları karşılaştırmalı liderlik tablosuna sokmaz. Farklı veri bölmeleri, sınıf eşleştirmeleri ve metrik uygulamaları ile sonuçlar doğrudan kıyaslanamaz.

## 4.2. 2025–2026'dan çıkan araştırma temaları

**Afetler arası genelleme:** Aynı afetin benzer coğrafi görüntülerinde yüksek skor alınması, yeni bir afet ve yeni kent dokusunda performans garantilemez. 2026 coğrafi yanlılık çalışması, dağılım-dışı coğrafyalarda bozulmayı incelemiş ve genellemenin ayrı değerlendirilmesi gerektiğine işaret etmiştir [S9].

**Dengesiz sınıflar ve hata maliyeti:** Hasarlı sınıflar az olabilir; özellikle ağır hasarı “hasarsız” diye değerlendirme türü hatalar farklı maliyet taşır. 2026 maliyet duyarlı çalışma, sınıf dengesizliği ve olaylar arası aktarım üzerinde yoğunlaşır [S10]. AERIS'te resmi score yanında sınıf bazlı recall ve hata matrisi raporlanmalı.

**Pikselden bina örneğine geçiş:** Kimi sistemlerin segmentasyon başarımı ile bina sayısı/örnek başarımı aynı doğrulukta ilerlemiyor. Insta-BDA, örnek-bazlı değerlendirme ve sayım sorununu araştırmıştır [S11]. Bu nedenle AERIS yalnızca renkli maske göstermemeli; bina bazlı özetler ve birleşik bina maskeleri gibi hataları da raporlamalı.

**Optik/SAR tamamlayıcılığı:** Bulut veya duman optik görüntüyü bozabilir. BRIGHT, SAR'ın farklı koşullarda araştırma olanağını gösterir [S12]. Ancak bu, zaman kısıtlı ilk sürüme doğrudan eklenmesi gereken bir özellik değildir.

## 4.3. Yenilik iddiası nasıl sınırlandırılmalı?

“İlk kez afet hasarı tespit eden yapay zekâ” ifadesi yanlış olur; bu araştırma alanı en az 2019'dan beri açıkça mevcut. “Dünyanın en iyi modeli” iddiası ancak aynı resmî test protokolünde kapsamlı kıyaslamayla yapılabilir.

AERIS için **portföy düzeyinde teknik özgünlük** şu kombinasyonda olabilir: (i) olay bazlı veri ayrımı, (ii) küçük GPU bütçesinde basamaklı modeller, (iii) belirsizlik nedeniyle seçici tahmin, (iv) hem piksel hem bina bazlı değerlendirme, (v) doğruluk ve gecikmenin aynı raporda verilmesi. Bu bileşenlerin tek tek literatürde var olduğu kabul edilmelidir; katkı, bunların şeffaf biçimde uygulanması ve ölçülmesidir.

# 5. Olası sistem mimarileri ve karar matrisi

## 5.1. Dört teknik alternatif

**A — Bina patch sınıflandırma:** Gerçek bina poligonlarından kesilen öncesi/sonrası RGB parçalarına bir sınıflandırıcı uygulanır. *Avantaj*: en hızlı başlangıç, küçük GPU ile uygulanabilir. *Zayıflık*: gerçek bina yerelleştirme çözülmez; “oracle building footprint” gerektirir. **Araştırma baseline'ı** olarak kullanılmalı.

**B — İki aşamalı sistem:** İlk model bina maskesi üretir; ikinci model maskelerden ya da crop'lardan hasar tahmini yapar. *Avantaj*: modüler, hata kaynağı ayrıştırılabilir, tek tek iyileştirme yapılır. *Zayıflık*: yerelleştirme hatası sınıflandırmaya taşınır. **İlk gerçek AERIS sistemi için önerilen yol.**

**C — Birleşik Siamese U-Net:** Öncesi ve sonrası görüntüyü ortak/ayrı encoder ile işleyip değişimden hasar haritası çıkarır. *Avantaj*: pre/post ilişkisini doğrudan öğrenebilir, uçtan uca sonuç verir. *Zayıflık*: eğitim, etiket hizalaması ve bellek talebi daha hassas. **İkinci ciddi deney olarak önerilen yol** [S6].

**D — Büyük foundation/transformer füzyonu:** SAM türevi segmentasyon ve modern değişim algılama ağlarının füzyonu. *Avantaj*: yüksek potansiyel, güncel literatür. *Zayıflık*: 4 GB VRAM, kısa takvim, tekrarlanabilirlik maliyeti. **Şimdilik araştırma kıyası; varsayılan uygulama değil** [S11].

## 5.2. Nitel karar matrisi

Puanlar **araştırmacı değerlendirmesidir; deney sonucuna dayanmaz**. 5 yüksek/olumlu demektir.

| Kriter | A: Patch | B: İki aşama | C: Siamese | D: Foundation |
|---|---:|---:|---:|---:|
| Çalışan ilk prototipe ulaşma | 5 | 4 | 2 | 2 |
| 4 GB GPU ile uygulanabilirlik | 5 | 4 | 3 | 1 |
| Gerçek uçtan uca iş akışı | 1 | 5 | 5 | 5 |
| Hata kaynaklarını açıklayabilme | 4 | 5 | 3 | 2 |
| Deneysel derinlik potansiyeli | 3 | 5 | 5 | 5 |
| Proje zaman riski düşük | 5 | 4 | 2 | 1 |

**Seçim:** A'yı yanlış kıyaslanmayan bir başlangıç modeli olarak kullan; B'yi çalışan ana sistem olarak kur; zaman ve donanım elverirse C ile anlamlı bir karşılaştırma yap. D'yi önceliklendirme.

## 5.3. Önerilen araştırma mimarisi

```text
[xBD indir / lisans kontrolü]
              |
[Veri manifesti + pre/post eşleştirme + olay-bazlı split]
              |
         +----+-----+
         |          |
 [Bina segment.] [Bina crop / oracle baseline]
         |          |
 [Tahmini bina maskesi]   |
         +----+-----+
              |
 [Pre/post hasar sınıflandırma]
              |
 [Kalibrasyon + seçici tahmin / 'incele']
              |
 [Piksel F1 / bina F1 / olay OOD / gecikme]
              |
 [Rapor + hata galerisi + güvenli inceleme demosu]
```

**Katman ayrımı:** `datasets/`, `models/`, `training/`, `evaluation/`, `inference/`, `reports/`, `demo/` ve `tests/`. Kod yapısının amacı gösteriş değil, deneylerin yeniden çalıştırılabilmesi.

# 6. Araştırma hipotezleri ve deney tasarımı

Aşağıdaki H1–H5, yapılmış bilimsel bulgular **değil**, araştırmayı yönlendirecek test edilebilir hipotezlerdir.

| Kod | Hipotez | Minimum kontrollü karşılaştırma | Birincil ölçüm |
|---|---|---|---|
| H1 | Pre/post bilgi kullanan model, yalnızca post kullanan modele kıyasla bazı olaylarda daha başarılı olabilir | Aynı backbone, aynı veri split'i, tek vs çift zaman | Olay-bazlı macro-F1 / damage-F1 |
| H2 | Sınıf ağırlıklı kayıp, nadir hasar sınıflarının recall'ını yükseltebilir | Normal CE vs weighted CE/focal; sabit seed | Sınıf F1/recall, hata dengesi |
| H3 | Afet-bazlı ayrım, rasgele patch ayrımına göre daha gerçekçi ve genellikle daha düşük skor verebilir | Rastgele patch vs event-held-out | OOD F1 düşüşü |
| H4 | Düşük güvenli tahminleri incelemeye ayırmak kalan otomatik tahminlerin doğruluğunu artırabilir | Eşik yok vs confidence threshold | Risk–coverage ve kalibrasyon |
| H5 | Daha hafif model, sınırlı F1 kaybıyla belirgin gecikme/bellek kazancı sağlayabilir | İki encoder/model boyutu | F1, p50/p95, peak VRAM |

Her deney için: hipotez, sabit split hash'i, seed, donanım/çerçeve sürümü, eğitim süresi, hiperparametreler, veri dönüşümleri, checkpoint ölçütü ve ham metrik dosyaları yazılmalı. En az iki seed yararlı; zaman yetmiyorsa tek seed sonucunun sınırı açıkça belirtilmeli.

## 6.1. Veri sızıntısına karşı protokol

**Kesin kural:** Aynı kaynak uydu sahnesinden, aynı yakın coğrafi tile'dan ve aynı öncesi/sonrası çiftinden türeyen patch'ler eğitim ile test arasında bölünmemeli. Aksi durumda model yeni bir yer öğrenmek yerine yakın komşu görsel dokuları ezberleyebilir.

İki ayrı değerlendirme önerilir:

- **ID (in-distribution) test:** Benzer afet/alan dağılımında ancak kaynak görüntü grupları bağımsız örnekler.
- **OOD (out-of-distribution) test:** Seçilmiş en az bir afet olayı veya coğrafya tamamen dışarıda bırakılır. Olay sayısı ve sınıf kapsamı raporlanır.

TorchGeo'nun `xBDDistShift` veri sınıfı, afetlere göre bir dağılım kayması örneği sunuyor; **fakat varsayılan biçimi hasar sınıflarını ikili bina maskesine indirger. Dört hasar sınıfı için doğrudan hazır OOD protokolü sanılmamalıdır** [S3].

**Not:** Resmî challenge skoruyla olay-bazlı genelleme puanı farklı test protokolleridir; raporda ayrı tutulmalı.

## 6.2. Sınıf dengesizliğini ele alma

İlk sırada veri dağılımını ölç; sonra farklı müdahaleleri ablation ile dene:

- Uniform sampling karşısında sınıf dengeli / olay dengeli örnekleme.
- Cross-entropy karşısında weighted cross-entropy ve, uygun olduğunda, focal loss.
- Aşırı minor/major confusion için sınıf bazlı hata analizi.
- Geometrik ve fotometrik augmentasyonlar; öncesi ve sonrası görüntüde **uyumlu geometrik dönüşüm** uygulanması.

Oversampling az sınıflar için eğitim başarımını artırırken genellemeyi bozabilir; bu nedenle yöntemi değil, **ölçülmüş değişimi** raporla. Hasar ölçeği ordinal yapıda olsa da doğrudan ordinal regression seçimi ayrı deney gerektirir.

# 7. Başarı metrikleri: skor hesabı ve yanlış kıyaslardan kaçınma

Resmî xView2 değerlendirme kodu toplam puanı aşağıdaki gibi tanımlar [S15]:

**`Score = 0.3 × F1_localization + 0.7 × F1_damage`**

Dört hasar sınıfının F1 değerleri `F1_no`, `F1_minor`, `F1_major`, `F1_destroyed` olsun:

**`F1_damage = 4 / (1/F1_no + 1/F1_minor + 1/F1_major + 1/F1_destroyed)`**

Uygulamada resmî kod sıfıra bölme için küçük epsilon kullanır. Bu harmonik ortalama nadir bir sınıftaki düşük başarımı güçlü biçimde cezalandırır. Dolayısıyla “%90 accuracy” gibi tek bir sayı tek başına yeterli değildir.

**Zorunlu metrik paketi:**

1. Bina yerelleştirme: pixel-level precision/recall/F1 ve IoU; mümkünse bina bazlı F1 ayrı.
2. Hasar: dört sınıf için precision, recall ve F1; confusion matrix; macro-F1, xView2 damage-F1.
3. Uçtan uca: resmî skor uyumlu değerlendirme; oracle bina konumlu sonucun ayrı raporlanması.
4. OOD: olay bazında skorlar, ağırlıklı ve ağırlıksız ortalamalar, en kötü olay performansı.
5. Güvenilirlik: calibration/ECE (uygun etiket düzeyinde), reliability diagram, risk–coverage eğrisi, ret edilen örnek oranı.
6. Sistem: tek görüntü çifti ve görüntü patch'i başına süre, p50/p95, CPU/GPU, batch, çözünürlük, donanım, peak VRAM/RAM.

**Önemli analiz örneği:** Bir sistem tüm hasarsız binaları çok iyi bulup “minor damage”ı neredeyse hiç ayırt edemeyebilir. Yüksek doğruluk görünüşte etkileyici olsa da resmi hasar F1 düşük kalır. Bu sınıf için örnek bazlı hata galerisi ve belirsizlik analizi göstermek daha dürüsttür.

## 7.1. Kalibrasyon ve abstention

Modelin `softmax` olasılığı otomatik olarak doğru güven olasılığı değildir. ECE veya güvenilirlik diyagramı ile kalibrasyon gözlenmeli; gerekiyorsa yalnızca validation üzerinde temperature scaling araştırılmalı. “Bilmiyorum, incelemeye gönder” seçeneği rapora üç sayı eklemelidir: otomatik karar kapsamı, bu kapsamda hata, ertelenen örnek sayısı. Yalnızca seçili örneklerdeki skoru gösterip reddedilenleri gizlemek metodolojik açıdan yanlıştır [S16].

# 8. Donanım, eğitim maliyeti ve uygulanabilirlik

Araştırma için varsayılan geliştirici makinesi **RTX 3050 Ti Laptop GPU (4 GB VRAM), 32 GB RAM, 1 TB NVMe, Linux** olarak ele alınmıştır. Bu donanım sınırlı ama doğru kapsamla yeterli bir prototip için değerlendirilebilir.

## 8.1. 4 GB VRAM stratejisi

- **Model:** İlk denemede MobileNetV3 / ResNet18 gibi hafif encoder veya küçük U-Net; ImageNet ön-eğitimli ağırlıklar kullanılabilir.
- **Giriş:** 128×128/256×256 patch ile başla; büyütmenin etkisini ölçmeden 1024×1024 tam görüntü eğitimi yapma.
- **Batch:** VRAM'e göre küçük batch; gerekirse gradient accumulation.
- **Precision:** GPU ve kurulu PyTorch sürümüyle test edilerek AMP/FP16; her cihazda BF16 destekli varsayma.
- **Transfer learning:** Önce encoder dondurup yalnızca classifier head eğitimi; ardından kademeli fine-tuning.
- **Memory:** `torch.cuda.max_memory_allocated`, kaynak monitörü, dataloader workers ve cache tüketimini kaydet.
- **IO:** Ham ve işlenmiş veri boyutunu, eğitim cache'ini ve model ağırlıklarını ayrı dizinlerde tut.

**Eğitim süresi taahhüdü verilmemelidir:** Gerekli saatler mimariye, veri sayısına, batch'e, CPU/disk hızına ve GPU güç limitine bağlıdır. İlk 100–500 batch ile gerçek throughput ölçülmeden “3 saatte eğitim tamam” denmemelidir.

## 8.2. ONNX ve benchmark yaklaşımı

PyTorch'un ONNX aktarımı ve ONNX Runtime, ayrı çıkarım arka uçlarının denenmesine olanak verir [S17]. Fakat ONNX kullanımı **otomatik hızlanma garantisi değildir**. Aynı görüntü boyutu ve ön/son işleme zinciriyle, warm-up sonrası p50/p95 gecikmeler, doğruluk tutarlılığı ve bellek ölçülmelidir. CUDA/cuDNN sürümü uyumluluğu bağımlılık kilidinde doğrulanmalıdır.

**Donanım açısından karar:** İlk büyük başarı hedefi SOTA model değil, yerelde defalarca yeniden eğitilip sonuçları kontrol edilebilen hafif modeldir.

# 9. Teknik ürünleşme araştırması: demo neyi kanıtlamalı?

AERIS'in kullanıcı arayüzü bir “görsel şov” değil, modelin incelenebileceği teknik bir araç olmalı. Önerilen araştırma demosu aşağıdakileri göstermeli:

- Öncesi/sonrası uydu görüntüleri için kilitli karşılaştırma (slider veya yan yana).
- Tahmin edilen bina maskesi ve görünür hasar sınıflarının renkli overlay'i.
- Seçili bir bina için tahmin etiketi, güven düzeyi ve görüntü parçası.
- “İnsan incelemesi gerekli” etiketi ve modelin neden kararsız kaldığına yönelik hesaplanabilir sinyaller.
- Sınıf dağılımı ve özellikle **yanlış negatif ağır hasar örnekleri** galerisi.
- Tek örnek tahmini ile toplu benchmark modunun açıkça ayrılması.
- “Bu bir araştırma prototipidir; sahada güvenli hasar tespitinin yerine geçmez” uyarısı.

Arayüz için Streamlit/Gradio tarzı hafif demo, zaman tasarrufu sağlar; web geliştirmeyi merkeze taşımaz. Nihai teknoloji seçimi ürün planı belgesinde yapılmalı. Sunucu dağıtımı yerine indirilebilir yerel demo, lisans ve VRAM kısıtları bakımından daha kontrollü olabilir.

## 9.1. GitHub'da ispat değeri yüksek kanıtlar

| Artefakt | Değerlendirene neyi ispatlar? |
|---|---|
| Çalışan CLI / örnek demo | Sistem başkasının makinesinde çalışabilir |
| Data card + lisans notu | Veri etiği ve veri mühendisliği |
| `splits.json` ve kaynak ID mantığı | Veri sızıntısı bilinci |
| Model card + `limits.md` | Güvenilirlik ve sorumlu iletişim |
| Deney karşılaştırma tablosu | Araştırma yöntemi ve ablation disiplini |
| OOD test raporu | Gerçek dünya genellemesi anlayışı |
| `pytest` + CI | Yazılım mühendisliği pratiği |
| CPU/GPU/ONNX benchmark | Performans ve sistem yaklaşımı |
| 90–120 saniyelik teknik demo | Teknik çıktıyı özlü anlatabilme |

**Dürüstlük kuralı:** Hazır pretrained ağırlıklar, başka bir repo kodu ve dış kütüphaneler açıkça atıflanmalı. Kopyalanan benchmark skorları yeni elde edilmiş AERIS skorları gibi sunulmamalı.

# 10. Kritik risk kaydı

| Risk | Etki | Olasılık (öngörü) | Araştırmadan çıkan azaltma |
|---|---|---|---|
| xBD erişim/kullanım koşulları beklenmedik | Yüksek | Orta | Önce indirme ve lisans doğrulama; ham veriyi depoya koymama |
| Sınıf dengesizliği / minor sınıf başarısızlığı | Yüksek | Yüksek | Sınıf F1 ve hata galerisi; loss/sampling ablation |
| Patch bazlı veri sızıntısı | Çok yüksek | Orta | Olay, sahne ve tile grup manifestleri |
| 4 GB VRAM'de OOM | Yüksek | Yüksek | Hafif backbone, 128–256 px, küçük batch |
| Uydu öncesi/sonrası hizalama hatası | Orta–yüksek | Yüksek | Eşleşme kontrolleri, görüntü hizalama analizi |
| Uçtan uca modelin çok uzaması | Yüksek | Orta | Oracle patch baseline + kademeli iki aşamalı sistem |
| Gösterişli UI'nin model araştırmasını bastırması | Orta | Yüksek | Demo araştırma kanıtlarından sonra |
| Model puanının fazla iddia edilmesi | Yüksek | Orta | Her metrik için veri protokolü ve limit etiketi |
| Büyük model / SAR genişlemesinin takvimi bozması | Orta | Yüksek | BRIGHT/SAM yalnızca opsiyonel araştırma hattı |

Buradaki olasılık değerlendirmeleri **ölçülmüş frekanslar değil, proje planlama varsayımlarıdır**. Ürün planında gereksinimler ve kontrol noktaları netleştirilmeli.

# 11. Araştırma sonucunda tavsiye edilen kapsam sınırı

## 11.1. Birinci teslim için vazgeçilmez araştırma çıktıları

- xBD kaynak, lisans ve dosya yapısının doğrulanması.
- Dört hasar sınıfı ve arka plan ayrımının hatasız veri temsili.
- Basit ama yeniden üretilebilir iki basamak: bina segmentasyonu + hasar sınıflandırma.
- Başlangıç modeli karşısında geliştirilmiş modelin *aynı split ve metrikle* kıyaslanması.
- En az bir olay-bazlı ayrı test; karşılaştırma maliyeti açıklanmış olmalı.
- Sınıf bazlı precision/recall/F1, birleşik xView2 uyumlu skor ve confusion matrix.
- Başarısız örnekler ve “bu model ne zaman kullanılmamalı?” anlatısı.
- Çalışan gösterim ve teknik README.

## 11.2. Zaman kalırsa değerli ama başlangıç için zorunlu olmayanlar

- Ayrı Siamese mimari deneyleri.
- Kalibrasyon ve risk–coverage karşılaştırmasının daha kapsamlı sürümü.
- Farklı afet türlerine göre sistematik leave-one-event-out analizi.
- ONNX/TensorRT gibi daha derin hızlandırma araştırmaları.
- BRIGHT ile çapraz-modal genelleme denemeleri.
- Nesne bazlı instance segmentation ve değişen coğrafya üretken modelleri.

**Kesin öneri:** Proje üçlü portföyün yalnızca bir öğesi olduğundan, **çok sayıda yarım deney yerine küçük fakat metodolojik olarak düzgün bir AERIS çalışması** daha değerlidir.

# 12. İkinci araştırma için açık sorunlar ve özel inceleme listesi

Bu doküman 1. aşamadır. Derin araştırma aracına verilecek 2. aşamada **bu raporun tekrar yazılmasını değil, seçimleri doğrulayacak ve çürütebilecek kanıtların aranmasını** öneriyorum.

## A. Veri ve reprodüksiyon soruları

1. Kasım 2026 itibarıyla xBD indirmenin gerçekten çalışan kanalları, hesap şartları, resmî paket adları, her paketin güncel boyutu ve lisans maddeleri neler? Orijinal şartların ekran/bağlantı referansı nedir?
2. Dört hasar sınıfını koruyan ve olay-bazlı veri ayrımını destekleyen bakımlı PyTorch veri yükleyicileri hangileri? TorchGeo 0.10.x davranışındaki önemli noktalar neler?
3. 2024–2026 çalışmaları içinde tam kod + ağırlık + split dosyaları + yeniden üretilebilir metrikler sunan en az beş yöntem hangileri?
4. xView2 puanlama kodunun yerelleştirme/hasar etiketlerini nasıl beklediği ve bina/nesne ayrımlarının teknik ayrıntıları neler?

## B. Bilimsel ve modeller arası sorular

5. 2025–2026'da xBD üzerinde, aynı değerlendirme protokolüyle kıyaslanabilir **hafif** U-Net, Siamese, CNN ve transformer modellerinin gerçek F1/VRAM/parametre sonuçları neler?
6. “Minor damage” ve sınıf dengesizliği için controlled ablation sonucu en güvenilir kayıp/örnekleme seçimleri neler?
7. Coğrafi ve afet-bazlı veri kaçağını önleyen yayımlanmış protokollerin eksiksiz split manifestleri mevcut mu?
8. Güven puanı/abstention, calibration, selective classification literatüründe bina hasarı özelinde denenmiş iyi yöntemler neler?

## C. Fizibilite ve portföy soruları

9. RTX 3050 Ti **4 GB** GPU'da gerçek deney yapmış kamuya açık reprodüksiyonlar veya 4–6 GB VRAM profilleri var mı?
10. Optik RGB'den SAR/BRIGHT'a geçiş, bu başvuru takvimi için uygun mu, yoksa ayrı gelecek proje mi olmalı?
11. Veri lisansına uygun bir GitHub demosunda hangi görüntüleri/örnekleri yayımlamak güvenli?
12. BAYKAR 2027 Yapay Zekâ Yazılım Geliştirme rolü için model araştırması, C++ sistem yönü, görüntü işleme ve yazılım mühendisliği kanıtı nasıl en kuvvetli şekilde birlikte sergilenir? İlanla **doğrudan belgelenen** ve **çıkarım** olan iddiaları ayrı yaz.

**Derin araştırmadan beklenen çıktı:** Kaynak güvenilirliği derecelendirilmiş karşılaştırmalı yöntem tablosu; doğrulanmış veri/lisans erişim kontrol listesi; 4 GB VRAM için kanıtlı model seçimi; olası deney özgünlük açığı; uygulanabilirlik gerekçeleri ve bu R01'de değişmesi gereken kararlar.

# 13. Ön araştırma hükmü

**AERIS fikri güçlü, ama özgünlüğü “yıkılmış bina bulmak” değil.** Araştırmaya dönüşmesi için genellemenin, belirsizliğin, veri sızıntısının ve gecikmenin ölçülmesi gerekli. Benchmark'a uygun olmayan metrikleri yan yana sıralamak, yalnızca güzel overlay'ler göstermek ve tek bir rastgele veri bölmesiyle genel geçer başarı iddia etmek projeyi zayıflatır.

**Birinci teknik tercih:** xBD + modüler iki aşamalı hafif model + olay bazlı test + sınıf bazlı F1 + örnek-hata inceleme. **İkinci tercih:** aynı temelde daha güçlü Siamese karşılaştırması. **Öncelikli araştırma riski:** veri ve lisans erişimi. **En güçlü mülakat anlatısı:** Modelin güçlü olduğu kadar başarısız olduğu koşulları da gösterebilen, kendi test protokolünü kurmuş genç mühendis.

---

# Kaynakça ve doğrulama bağlantıları

**Erişim/araştırma tarihi:** 20 Kasım 2026. Makale bulguları ve erişim koşulları zamanla değişebilir. Kaynaklardan alınan rakamların kapsamı belgede belirtilmiştir.

[S1] BAYKAR Kariyer, *2027 Yaz Dönemi Staj — Yapay Zeka Yazılım Geliştirme*. https://kariyer.baykartech.com/tr/acik-pozisyonlar/detay/2027-yaz-donemi-staj-yapay-zeka-yazilim-gelistirme

[S2] US Defense Innovation Unit, *xView Computer Vision Challenge Series*. https://www.diu.mil/ai-xview-challenge

[S3] TorchGeo 0.10.1, *xBD Dataset Documentation*. https://docs.torchgeo.org/en/stable/api/datasets/xbd.html

[S4] Gupta, R. et al. (2019), *xBD: A Dataset for Assessing Building Damage from Satellite Imagery*. https://arxiv.org/abs/1911.09296 — erişilebilir HTML: https://arxiv.org/html/1911.09296v1

[S5] DIUx-xView, *xView2 Baseline* (resmî açık kaynak başlangıç modeli). https://github.com/DIUx-xView/xView2_baseline

[S6] Microsoft, *Building Damage Assessment — CNN Siamese*. https://github.com/microsoft/building-damage-assessment-cnn-siamese

[S7] Paul Borne ve çalışma ekibi, *xView2 Strong Baseline*. https://github.com/PaulBorneP/Xview2_Strong_Baseline

[S8] Hafner, S. et al. (2025), *DisasterAdaptiveNet: A robust network for multi-hazard building damage detection*. https://github.com/SebastianHafner/DisasterAdaptiveNet ; makale DOI: https://doi.org/10.1016/j.jag.2025.104756

[S9] *Geographic Bias Analysis and Cross-Domain Generalization in Deep Learning-Based Building Damage Assessment* (2026), Remote Sensing. https://www.mdpi.com/2072-4292/18/10/1529

[S10] *Framework for Cross-Disaster Building Damage Assessment Using Cost-Sensitive Learning* (2026), Remote Sensing. https://www.mdpi.com/2072-4292/18/17/2920

[S11] *Insta-BDA: Instance-Aware Building Damage Assessment and Counting via Foundation Model Fusion* (2026), Remote Sensing. https://www.mdpi.com/2072-4292/18/14/2347

[S12] Chen, H. et al. (2025), *BRIGHT: a globally distributed multimodal building damage assessment dataset*, Earth System Science Data. https://essd.copernicus.org/articles/17/6217/2025/ ; kod: https://github.com/ChenHongruixuan/BRIGHT

[S13] TerraFabric, *xBD Building Damage Dataset — Data Use & License Listing*. https://terrafabric.world/collections/xbd-building-damage

[S14] AWS Registry of Open Data, *Maxar Open Data Program*. https://registry.opendata.aws/digital-globe-open-data/

[S15] DIUx-xView, *xView2 Scoring Source Code* (resmî değerlendirme formülü). https://github.com/DIUx-xView/xView2_scoring/blob/master/xview2_metrics.py

[S16] NIST, *AI Risk Management Framework* ve güvenilirlik ölçütleri. https://www.nist.gov/itl/ai-risk-management-framework ; https://airc.nist.gov/airmf-resources/airmf/3-sec-characteristics/

[S17] ONNX Runtime, *Python Quickstart & CUDA Execution Provider*. https://onnxruntime.ai/docs/get-started/with-python ; https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

**Kaynak değerlendirmesi:** [S1–S5, S15] resmî ilan/veri/puanlama ana kaynakları; [S8–S12] araştırma yayınları; [S13–S14] lisans için yardımcı kayıtlar ve tek başlarına kesin hukuki güvence değil; [S16–S17] standart/teknik belgeler. Bilimsel makaleler arasındaki sonuçlar, değerlendirme protokolü eşitliği sağlanmadan performans sıralamasına dönüştürülmemiştir.
