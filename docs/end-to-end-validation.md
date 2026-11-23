# 29–30 Kasım uçtan uca değerlendirmesi

Hat, afet öncesi RGB görüntüden bina maskesi çıkarır. Sekiz komşuluklu bağlı bileşenler tespit edilip afet sonrası görüntüden eğitimle aynı kurallarla kırpılır. ResNet18’in dört sınıflı tahminleri adayın kendi piksellerine geri yazılır. Uçtan uca tahmin, görüntü çiftini girdi olarak alır. Değerlendirme tarafı gerçek bina poligonlarını eşleştirme ve oracle kıyası için ayrıca okur. [OpenCV bağlı bileşen yöntemi](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html) · [Çıkarım ayarları](../configs/inference.yaml)

Lokalizasyon checkpoint’inin 0,40 eşiği korundu. Validation maskelerinde en az 2 piksel en ve boy koşuluyla 1, 4, 8, 16 ve 32 piksel alan sınırları karşılaştırıldı. En yüksek piksel F1’ini veren 32 piksel seçildi. Alt sınır 1 iken 5.989, 32 iken 5.554 aday kaldı. Eşik taraması yerel `outputs/pipeline/nov30-component-sweep.json` dosyasında.

## Validation sonuçları

309 Hurricane Florence görüntüsünde her iki koşul aynı resmî hedef PNG’lerle puanlandı. Oracle koşulunda afet sonrası gerçek bina poligonları kullanıldı. Bina bazındaki eşleştirme de bu poligonlar üzerinde yapıldı. Piksel lokalizasyonu için arşivin afet öncesi hedef maskeleri, piksel hasarı için afet sonrası hedef maskeleri kullanıldı. [G0 veri bölmesi](g0.md#dondurulan-id-ve-ood-bölmesi)

| Piksel metriği | Tahmin edilen bina adayları | Gerçek bina poligonları |
|---|---:|---:|
| Harmonik hasar F1 | 0,0898 | 0,1177 |
| Hasar macro-F1 | 0,2546 | 0,3347 |

Uçtan uca lokalizasyon F1’i 0,7779, IoU 0,6366 ve birleşik xView2 uyumlu skor 0,2962 oldu. İki koşulun 309’ar görüntülük PNG çıktıları resmî puanlayıcıyla tekrar puanlandı. Yedi referans skor alanı iki koşulda da eşleşti. [Resmî xView2 puanlayıcısı](https://github.com/DIUx-xView/xView2_scoring/blob/ea0793da6f66a71236f2c4a34536d51beff483ab/xview2_metrics.py)

5.770 gerçek binanın 3.561’i bire bir eşleşti ve yakalama oranı %61,7 oldu. 330 birleşik aday, 784 farklı gerçek binayı kapsadı. Eşleşmesiz 1.993 aday ve 2.209 gerçek bina vardı. Birleşme sayımı bire bir eşleştirmeden bağımsızdır.

| 1.299 ağır hasarlı veya yıkılmış bina | Adet |
|---|---:|
| Bulunamadı | 301 |
| Birleşik adayda kaldı | 172 |
| Hasarsız tahmin edildi | 661 |
| Hafif hasarlı tahmin edildi | 58 |
| Ağır hasarlı tahmin edildi | 28 |
| Yıkılmış tahmin edildi | 79 |

Tabloda her gerçek ağır hasar binası bir kez sayılır. Birleşik aday kapsamı, bulunamama ve sınıf atamasından önce uygulanır. Sentetik entegrasyon testi tam 0,5 IoU sınırını, eşitlikte kimlik sırasını, birleşme önceliğini, eşleşmeyen adayı ve sınıflandırılmamış gerçeğin lokalizasyona katılmasını doğruladı. Etiketsiz bir gerçek görüntü çiftiyle CPU çıkarımı çalıştırıldı. PyTorch ortamında 89 test geçti. [Eşleştirme testleri](../tests/unit/test_building_evaluation.py) · [Entegrasyon testi](../tests/integration/test_pipeline.py)

## Tekrar çalıştırma

```bash
.venv-ml/bin/python -m aeris evaluate-pipeline --device cuda \
  --output-dir outputs/pipeline/nov30-repeat
```

Bu koşunun yerel dizini `outputs/pipeline/nov30-validation/`. `predictions/`, `oracle_predictions/` ve `targets/` dosyaları xView2 puanlayıcısının beklediği biçimdedir. `buildings.jsonl` görüntü başına adayları, ham skorları, eşleştirmeleri ve hata sayımlarını içerir. `metrics.json` iki koşulun skorlarını, checkpoint SHA-256 değerlerini ve bölme özetini kaydeder.
