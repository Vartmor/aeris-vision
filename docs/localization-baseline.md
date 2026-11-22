# 26 Kasım lokalizasyon modeli

ResNet18 encoder’lı U-Net, ImageNet ağırlıklarından başlayarak dondurulmuş train bölümünden seed 2026 ile seçilen 320 görüntü üzerinde eğitildi. Alt küme dokuz eğitim sahnesini içeriyor. Eğitimde 5.120 adet 256 × 256 yama, validation’da Hurricane Florence’ın ayrılmış sahnesinden 4.944 yama ve 309 tam görüntü kullanıldı. Girdi afet öncesi RGB, hedef resmî ikili xView2 lokalizasyon maskesi. [Veri bölmesi](g0.md#dondurulan-id-ve-ood-bölmesi) · [Model konfigürasyonu](../configs/localization.yaml)

Her epoch’ta dokuz olasılık eşiğinin TP, FP ve FN sayıları validation piksellerinde toplandı. En yüksek F1’e sahip epoch ve eşik kaydedildi. F1 eşitliğinde 0,5’e en yakın eşik, uzaklık da eşitse küçük eşik seçilir. Epoch F1’leri eşitse ilk checkpoint korunur.

## Validation sonucu

| Ölçüm | Değer |
|---|---:|
| Seçilen epoch | 5 / 5 |
| Bina olasılık eşiği | 0,40 |
| Piksel F1 | 0,777836 |
| Piksel IoU | 0,636441 |
| F1, aynı checkpoint ve 0,50 eşikle | 0,776493 |
| Doğru pozitif piksel | 4.523.529 |
| Yanlış pozitif piksel | 1.137.760 |
| Yanlış negatif piksel | 1.446.248 |

Bu sonuçlar checkpoint ve eşik seçiminde kullanılan tek Hurricane Florence sahnesine ait. Küçük bileşen filtresi bu piksel ölçümüne uygulanmadı.

Yerel koşu dizini `outputs/localization/nov26-baseline/`. `best.pt` seçilen ağırlıkları ve eşik değerini, `metrics.json` alt küme kimliklerini, epoch ve eşik ölçümlerini ve maskelerin piksel sayımlarını içerir. Checkpoint SHA-256 değeri `e70486b8cdc33bced6982994a789a3f5205764c5262c02ea9d3f44faf461dd35`.

Checkpoint yeniden yüklenerek `validation_masks/` dizinine 309 adet 1.024 × 1.024 ikili maske yazıldı. PNG’ler tekrar açılıp gerçek hedeflerle karşılaştırıldığında piksel sayımları ve metrikler birebir eşleşti. `validation_preview.png`, üç örnekte gerçek maskeyi, tahmini ve hata piksellerini gösterir.

RTX 3050 Ti üzerinde beş epoch’un medyan eğitim süresi 67,36 saniye, PyTorch’un tepe bellek rezervi 288 MiB oldu. PyTorch ortamında 83 test geçti.

## Tekrar çalıştırma

```bash
.venv-ml/bin/python -m aeris train --task localization \
  --config configs/localization.yaml --device cuda --train-tiles 320 --epochs 5 \
  --thresholds 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 \
  --validation-masks --cache-dir data/cache/localization \
  --run-dir outputs/localization/nov26-repeat
```
