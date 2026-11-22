# AERIS VISION

Afet öncesi ve sonrası uydu görüntülerinden binaları ve görünür hasarı incelemek için geliştirilen bir araştırma projesi. İlk geliştirme aşamasında Python paketi, konfigürasyon dosyaları ve veri indirmeden çalışan testler hazır.

## Yerel kurulum

Python 3.11 veya üstü ve `uv` ile proje dizininde çalıştır.

```bash
uv sync --locked
uv run --locked aeris --help
uv run --locked python -m aeris config check --config configs/data.yaml
uv run --locked pytest
```

`config check` bir YAML dosyasının okunabildiğini ve kökte metin anahtarları olan bir eşleme içerdiğini doğrular. Aşama bazlı konfigürasyon kuralları, ilgili veri ve model komutlarıyla birlikte eklenecek.

## Veri denetimi

xView2 **Challenge training set** paketini kaydından sonra `data/downloads/` dizinine indir. `data prepare` arşivi doğrulayıp `data/raw/xbd/` altına açar ve SHA-256 kaydını bırakır. Paket zaten bu dizine açıldıysa doğrudan `data audit` çalıştır. [xView2 veri portalı](https://xview2.org/dataset) · [G0 veri kaydı](docs/g0.md)

```bash
uv run --locked aeris data prepare --config configs/data.yaml --archive data/downloads/train_images_labels_targets.tar.gz
uv run --locked aeris data audit --config configs/data.yaml --check-targets
uv run --locked aeris data verify --config configs/data.yaml
```

Komut, konfigürasyondaki yolları çalıştırıldığı dizine göre çözer ve iki dosya yazar:

- `data/processed/manifest.jsonl`: Her görüntü çifti için bir kayıt. Kayıtta olay, sahne ve mekânsal grup, dosya yolları ve SHA-256 değerleri, çekim metadatası, hasar sınıfı sayıları, `validation_status` ve sorun kodları bulunur.
- `data/processed/audit.json`: Olay bazında sınıf desteği, sorun sayıları, metadata kapsamı, gruplama özeti ve manifest hash'i. `--check-targets` verildiğinde AERIS maskelerinin paketteki resmî target PNG'leriyle uyumu da raporlanır.

Bir pikselin binaya ait sayılması için merkezinin poligonun içinde ya da sınırında olması gerekir. Hasar etiketi eksik, tanımsız veya `un-classified` olan binalar 255 (ignore) değerini alır; hiçbiri `no damage` sayılmaz.

## Veri bölme

`data audit` çalıştıktan sonra veri bölmesini üret ve doğrula.

```bash
uv run --locked aeris split make --config configs/data.yaml
uv run --locked aeris split verify --config configs/data.yaml
```

Proje konfigürasyonu Hurricane Michael'ı OOD olarak, incelenmiş sahne gruplarını da validation, kalibrasyon ve ID test için sabitler. Bu atamalarda her ID değerlendirme olayı farklı bir grupla eğitimde de temsil edilir. ID bölümlerinde sahne, mekânsal grup ve mevcut görüntü SHA-256 bağlantıları birlikte tutulur. OOD olayıyla bir grubu paylaşan diğer olay örnekleri varsa dağıtım dışı olarak raporlanır. [Bölme gerekçesi ve sınıf desteği](docs/g0.md#dondurulan-id-ve-ood-bölmesi)

`data/processed/splits.json` dosyası örnek ve grup üyeliğini, sınıf desteğini, manifest ve bölme hash'lerini kaydeder. `test` ID testini, `ood_test` eğitimden tamamen ayrılmış olayı gösterir. Eğitim ve doğrulamada dört hasar sınıfının da desteği aranır. Aynı girdiyle komut tekrar çalıştırılabilir. Manifest veya konfigürasyon değiştiğinde yeni bir çıktı yolu kullanılır.

## Piksel değerlendirmesi

xView2 biçimindeki tahmin ve hedef PNG dizinlerini puanlamak için çalıştır.

```bash
uv run --locked aeris evaluate --predictions outputs/predictions --targets outputs/targets --output outputs/scores.json
```

Her görüntü için 1024×1024, 8 bitlik lokalizasyon ve hasar maskesi gerekir. Dosya adları `test_localization_00001_prediction.png` ve `test_damage_00001_prediction.png` biçimindedir. Hedef dosyaları `_target.png` ile biter. PNG değerleri 0–4 aralığındadır. Sınıflandırılmamış bina pikselleri hasar hedefinde 0 olarak yazılır ve lokalizasyon hedefinde bina olarak kalır. JSON, xView2 uyumlu yerel skoru, sınıf desteğini ve piksel karışıklık matrisini içerir.

## Lokalizasyon eğitimi

Dondurulmuş veri bölmesi ve PyTorch ortamı hazırsa 32 eğitim yamasıyla GPU smoke ve overfit testi çalıştırılabilir. Koşu için yeni bir yerel çıktı dizini kullan. [PyTorch kurulumu](docs/environment.md) · [25 Kasım ölçümleri](docs/localization-smoke.md) · [26 Kasım modeli](docs/localization-baseline.md)

```bash
.venv-ml/bin/python -m aeris train --task localization \
  --config configs/localization.yaml --device cuda --tiny --epochs 48 \
  --cache-dir data/cache/localization --run-dir outputs/localization/tiny-repeat
```

`--tiny` eğitim bölümündeki aynı 32 yamada ölçüm yapar. Bu seçenek çıkarıldığında komut train yamalarında eğitir ve validation F1 ile checkpoint seçer. `--train-tiles` seed’e bağlı bir train alt kümesi seçer. `--thresholds` validation eşiklerini birlikte değerlendirir. `--validation-masks` seçilen checkpoint’in tam görüntü maskelerini üretir. Koşu dizininde `metrics.json` ve `best.pt` bulunur.

## Geliştirme

- `configs/` veri, eğitim ve değerlendirme için v1.1 başlangıç ayarlarını içerir.
- `src/aeris/` içinde veri, bölme, model, eğitim, değerlendirme ve çıkarım modülleri için paket yapısı bulunur.
- `tests/` konfigürasyon okuma, komut satırı ve kurulum kontrollerini içerir.
- İndirilen xBD arşivleri `data/downloads/`, açılmış veri `data/raw/`, yerel model ağırlıkları `checkpoints/` altında tutulur.

[Geliştirme planı v1.1](docs/geli%C5%9Ftirme%20plan%C4%B1%20v1.1.md) · [Yerel ortam kaydı](docs/environment.md)
