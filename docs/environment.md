# Yerel geliştirme ortamı

21 Kasım 2026 · Proje iskeleti

| Bileşen | Doğrulanan ortam |
|---|---|
| İşletim sistemi | LMDE 7 (gigi), x86_64 |
| Python | Yerel varsayılan 3.13.5. Ek kurulum testi 3.11.17. |
| Git | 2.47.3 |
| uv | 0.13.1 |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096 MiB |
| NVIDIA sürücüsü | 550.163.01 |

`uv sync --locked` proje sanal ortamını bağımlılık kilidinden kurar. 22 Kasım 2026'da Python 3.13.5 ve 3.11.17 ile 79 test geçti. İlk kurulumda wheel temiz bir Python 3.13 ortamında kuruldu ve `aeris --version` çalıştı.

## PyTorch doğrulaması

22 Kasım 2026'da Python 3.13.5 ile ayrı `.venv-ml` ortamı kuruldu. PyTorch `2.12.0+cu126` ve torchvision `0.27.0+cu126` resmî CUDA 12.6 deposundan kuruldu. [PyTorch sürüm listesi](https://pytorch.org/get-started/previous-versions/)

```bash
UV_PROJECT_ENVIRONMENT=.venv-ml uv sync --locked
uv pip install --python .venv-ml/bin/python --index https://download.pytorch.org/whl/cu126 'torch==2.12.0+cu126' 'torchvision==0.27.0+cu126'
.venv-ml/bin/python scripts/torch_smoke.py --output outputs/environment/torch-smoke.json
```

22 Kasım 2026'da CPU ve RTX 3050 Ti üzerinde ileri ve geri geçiş testleri geçti. İki cihazda da sonuç `14` ve gradyan `[2, 4, 6]` olarak doğrulandı. Test kaydı `outputs/environment/torch-smoke.json` konumunda.

İlk CUDA başlatması, `nvidia_uvm` modülü yüklüyken UVM aygıt düğümleri eksik olduğu için `CUDA_ERROR_UNKNOWN` verdi. Düğümler sistem terminalinde `sudo nvidia-modprobe -u -c 0` ile oluşturuldu ve test aynı oturumda geçti. [NVIDIA Linux kurulum kılavuzu](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

UVM modülü yüklüyken aygıt düğümleri eksikse sistem terminalinde `sudo nvidia-modprobe -u -c 0` çalıştır. Bu sürümde `-u` UVM modülünü yükler ve `-c 0` `/dev/nvidia-uvm` ile `/dev/nvidia-uvm-tools` düğümlerinin oluşturulmasını ister. İki düğümü `ls -l /dev/nvidia-uvm /dev/nvidia-uvm-tools` ile kontrol edip PyTorch testini yeniden çalıştır. CPU doğrulamasını tek başına çalıştırmak için `--device cpu` kullan. [Debian nvidia-modprobe 570.133.07 kaynak kodu](https://sources.debian.org/src/nvidia-modprobe/570.133.07-1/nvidia-modprobe.c/#L207)
