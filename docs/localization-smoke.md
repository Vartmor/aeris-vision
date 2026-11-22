# 25 Kasım lokalizasyon doğrulaması

ResNet18 encoder’lı U-Net, ImageNet başlangıç ağırlıkları ve BCE + Dice kaybıyla çalıştırıldı. RTX 3050 Ti üzerinde 256 × 256 yama, batch 2, AdamW 0,0001, FP16 AMP, iki veri yükleyici işçisi ve seed 2026 kullanıldı. [Eğitim konfigürasyonu](../configs/localization.yaml) · [Torchvision ResNet18 ağırlıkları](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)

G0 eğitim bölümünden 32 farklı afet öncesi görüntü seçildi. Her görüntüden bir yama alındı. 24 yamanın hedef maskesinde bina vardı, sekizi tamamen arka plandı. Eğitim girdisi RGB PNG, hedefi arşivin ikili `pre_disaster_target` maskesiydi. Seçilen 32 görüntü ve maskenin SHA-256 değerleri manifestle eşleşti. İki koşu da aynı yamaları kullandı. [Dondurulmuş G0 bölmesi](g0.md#dondurulan-id-ve-ood-bölmesi)

## Smoke ve aşırı öğrenme

| Ölçüm | Sonuç |
|---|---:|
| İlk önbellekli epoch | 32 yama, 16 forward/backward batch |
| İlk ve son eğitim kaybı | 1,485 → 0,950 |
| Başlangıç ve en iyi eğitim F1 | 0,000 → 0,931 |
| En iyi eğitim IoU | 0,870 |
| Seçilen epoch | 42 / 48 |
| En yüksek PyTorch bellek ayırımı | 244,2 MiB |
| En yüksek PyTorch rezervi | 288 MiB |

F1 ve IoU, eğitimde kullanılan aynı 32 yamanın pikselleri birleştirilerek ve 0,5 olasılık eşiğiyle hesaplandı. Bunlar aşırı öğrenme testi sonuçlarıdır. Checkpoint GPU’da yeniden yüklendi. TP 252.458, FP 25.411 ve FN 12.177 değerleri ile rapordaki F1 ve IoU yeniden elde edildi.

## Veri yükleme ölçümü

Önbellek, kaynak RGB ve hedef PNG’yi bir kez açıp tek bir uint8 NumPy dizisine yazar. Önbellek yolu manifest özetine, her dosya adı kaynak RGB ve hedef maske özetlerine bağlıdır. Önbellekli ve doğrudan PNG’den çıkarılan 32 yamanın görüntü ve maske tensörleri birebir eşleşti.

| Aynı yamalar ve batch ile ölçüm | PNG | Önbellek |
|---|---:|---:|
| Tek süreçte yama okuma ve hazırlama | 26,9 yama/sn | 530,3 yama/sn |
| Eğitim epoch medyanı | 0,605 sn | 0,351 sn |
| Epoch başına veri bekleme ve aktarım çağrısı medyanı | 0,252 sn | 0,017 sn |
| Eğitim hızı medyanı | 52,9 yama/sn | 91,2 yama/sn |

Epoch medyanları her koşunun 2–24. epoch’larını kapsıyor. Değerlendirme, checkpoint yazımı ve önbellek hazırlığı ayrı işlemler olarak ölçüm dışında tutuldu. 32 kaynak görüntünün önbellek hazırlığı 2,44 sn sürdü. xBD girdileri ve önbellek yerel `data/` dizininde saklanıyor.

## Yerel koşu kayıtları

- PNG koşusu `outputs/localization/nov25-tiny/metrics.json`.
- Önbellekli koşu `outputs/localization/nov25-tiny-cached/metrics.json`.
- Aşırı öğrenme checkpoint’i `outputs/localization/nov25-tiny-cached/best.pt`.
- Checkpoint SHA-256 `2fdcad2bab3ca177d911df56e0093ae96a35d3009a7875950f561fe6b6827aa8`.

PyTorch ortamında 81 test geçti. Sentetik CLI entegrasyon testi dondurulmuş bölmeden eğitim, validation ile checkpoint seçimi ve CPU’da yeniden yüklemeyi kapsıyor. [Çalışma ortamı](environment.md) · [Eğitim entegrasyon testleri](../tests/integration/test_localization_training.py)
