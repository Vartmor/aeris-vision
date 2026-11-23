# 27–28 Kasım hasar sınıflandırması

ImageNet ağırlıklı ResNet18, afet sonrası RGB görüntüden gerçek bina poligonlarıyla çıkarılan 160 × 160 parçaları işler. Görüntüye sığan poligon sınır kutusunun her kenarına %25 bağlam eklenir. Görünen kutunun iki kenarından biri 2 pikselin altındaysa, poligon geçersizse veya sınıf etiketi kullanılabilir dört sınıftan biri değilse örnek filtrelenir. En-boy oranı korunur ve kalan alan sabit RGB rengiyle doldurulur. Tahmin edilen bina adayları aynı kırpım fonksiyonuna girdi verebilir. [Kırpım kodu](../src/aeris/data/damage.py) · [Eğitim konfigürasyonu](../configs/damage_post.yaml)

İlk koşuda seed 2026 ile uygun 116.399 eğitim binasından 24.000 bina rastgele ve tekrarsız seçildi. Bunlar 1.066 görüntüye ve dokuz eğitim olayına ait. Ayrılmış Hurricane Florence sahnesindeki uygun 5.508 validation binasının tamamı değerlendirildi. [Dondurulmuş veri bölmesi](g0.md#dondurulan-id-ve-ood-bölmesi)

| Sınıf | Eğitim alt kümesi | CE ağırlığı | Validation |
|---|---:|---:|---:|
| No damage | 19.390 | 1,000 | 4.079 |
| Minor damage | 656 | 4,000 | 130 |
| Major damage | 1.885 | 3,207 | 1.245 |
| Destroyed | 2.069 | 3,061 | 54 |

Ağırlıklar yalnızca eğitime giren kırpımlardan `min(4, sqrt(n_max / n_c))` ile hesaplandı. Checkpoint seçimi bina başına bir etiketle ölçülen validation harmonik F1’e göre yapıldı. Eşitlikte macro-F1, ardından en erken epoch kullanıldı.

## Validation sonuçları

İlk koşuda hafif hasar F1’i düşük kaldı. İkinci koşuda sınıf başına en çok 6.000 bina seçildi. 3.149 uygun hafif hasar örneğinin tamamı ve diğer üç sınıftan 6.000’er örnek kullanıldı. Bu dağılımda hafif hasarın ağırlığı 1,380, diğerlerinin ağırlığı 1,000 oldu. Her iki koşu aynı validation kümesi ve checkpoint kuralıyla tamamlandı.

| Eğitim alt kümesi | Bina | Seçilen epoch | Harmonik F1 | Macro-F1 |
|---|---:|---:|---:|---:|
| Rastgele alt küme | 24.000 | 4 / 5 | 0,0545 | 0,2248 |
| Sınıf başına 6.000 üst sınır | 21.149 | 5 / 5 | 0,0769 | 0,2370 |

Harmonik F1’e göre ikinci koşunun checkpoint’i seçildi. Aşağıdaki matriste satırlar gerçek sınıfı, sütunlar tahmini gösteriyor.

| Gerçek sınıf | Hasarsız | Hafif | Ağır | Yıkılmış | Sınıf F1 |
|---|---:|---:|---:|---:|---:|
| Hasarsız | 3.201 | 181 | 242 | 455 | 0,7684 |
| Hafif | 103 | 12 | 8 | 7 | 0,0587 |
| Ağır | 918 | 85 | 53 | 189 | 0,0683 |
| Yıkılmış | 31 | 1 | 3 | 19 | 0,0525 |

Model 1.245 ağır hasarlı binadan 918’ini hasarsız tahmin etti. Metrikler checkpoint seçimi için kullanılan tek validation sahnesinde, gerçek bina poligonlarının sağlandığı koşula aittir.

Seçilen koşu `outputs/damage/nov28-class-cap/` altında. `best.pt` model ve kırpım ayarlarını, `metrics.json` epoch sonuçlarını ve karışıklık matrisini, `validation_logits.npz` 5.508 bina için ham skorları, gerçek sınıfı, bina kimliğini ve kırpım kutusunu içerir. Logit sütunları `no_damage`, `minor_damage`, `major_damage`, `destroyed` sırasındadır. Checkpoint SHA-256 değeri `40ffae49272fde7faf54f4fef5c9e7f1f1a8080a3f068dc11f5ee2d41d49fe6c`.

Checkpoint RTX 3050 Ti üzerinde tekrar yüklendi. 5.508 doğrulama örneğinin tamamı yeniden çalıştırıldı ve karışıklık matrisi ile sınıf metrikleri kaydedilen sonuçlarla birebir eşleşti.

## G1 ve G2 kontrolleri

32 gerçek bina ile yapılan CUDA aşırı öğrenme testinde üçüncü epoch’ta eğitim harmonik F1’i 1,000 oldu. PyTorch’un tepe bellek rezervi lokalizasyonda 288 MiB, seçilen hasar koşusunda 292 MiB ölçüldü. Hasar koşusunun medyan eğitim epoch’u 42,78 saniye sürdü. PyTorch ortamında 86 test geçti. [Lokalizasyon ölçümü](localization-baseline.md)

## Tekrar çalıştırma

```bash
.venv-ml/bin/python -m aeris train --task damage \
  --config configs/damage_post.yaml --device cuda --train-buildings 24000 \
  --epochs 5 --cache-dir data/cache/damage --run-dir outputs/damage/nov28-repeat

.venv-ml/bin/python -m aeris train --task damage \
  --config configs/damage_post.yaml --device cuda --max-crops-per-class 6000 \
  --epochs 5 --cache-dir data/cache/damage --run-dir outputs/damage/nov28-class-cap-repeat
```
