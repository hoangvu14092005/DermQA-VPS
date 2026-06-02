# DermNet VLM Eval — Docker Setup

Đóng gói VLMEvalKit + dataset DermNet để chạy eval trên bất kỳ máy nào có Docker và NVIDIA GPU.

## Yêu cầu

- Docker Desktop (Windows/Mac) hoặc Docker Engine (Linux)
- NVIDIA GPU + driver cập nhật
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (Linux)
  hoặc bật GPU trong Docker Desktop (Windows/Mac)
- Khoảng 30GB disk (image ~5GB + model HF cache ~3GB + ảnh DermNet)

## Cấu trúc

```
Dermnet-QA/
├── docker/
│   ├── Dockerfile              # build image
│   ├── run_eval.sh             # entrypoint script
│   ├── prepare_docker_dataset.py  # convert TSV path -> container path
│   ├── LMUData/                # TSV đã convert (path /app/data/...)
│   ├── hf_cache/               # cache model HuggingFace (volume)
│   └── outputs/                # kết quả eval (volume)
├── dermnet-output/dermnet-output/images/   # ảnh DermNet (mount vào container)
├── VLMEvalKit/                 # source code
└── docker-compose.yml
```

## Build image

```bash
docker compose build
```

Lần đầu sẽ tải base image PyTorch ~3GB và cài deps ~2GB. Build mất khoảng 10–15 phút.

## Chạy eval

```bash
docker compose up
```

Hoặc thiết lập tham số trên command line:

```bash
# Đổi model
MODEL=Qwen2.5-VL-7B-Instruct-AWQ docker compose up

# Chỉ chạy 1 dataset
DATA=DermNet_Test docker compose up
```

## Mang sang máy khác

Có 2 cách:

### Cách 1: Build lại trên máy đích (khuyến nghị)

```bash
git clone <repo>      # hoặc copy folder Dermnet-QA
cd Dermnet-QA

# Convert TSV (vì path Windows trong TSV gốc khác máy)
python docker/prepare_docker_dataset.py

docker compose build
docker compose up
```

### Cách 2: Save/load image

```bash
# Trên máy nguồn
docker compose build
docker save dermnet-vlmeval:latest -o dermnet-vlmeval.tar

# Copy file .tar + folder data sang máy đích, sau đó:
docker load -i dermnet-vlmeval.tar
docker compose up
```

## Volumes (mount)

| Host                                       | Container          | Mục đích                  |
|--------------------------------------------|--------------------|---------------------------|
| `./docker/LMUData`                         | `/app/data/LMUData`| File TSV (đã convert path)|
| `./dermnet-output/dermnet-output/images`   | `/app/data/images` | Ảnh DermNet               |
| `./docker/hf_cache`                        | `/app/data/hf_cache`| Cache model HuggingFace  |
| `./docker/outputs`                         | `/app/outputs`     | Kết quả eval (.xlsx)      |

## Kết quả

Sau khi chạy xong, file `.xlsx` xuất hiện ở:
```
docker/outputs/<MODEL>/T<timestamp>/<MODEL>_<DATASET>.xlsx
```

## HuggingFace token (tuỳ chọn)

Để tải model nhanh hơn (tránh rate limit), set token trước khi chạy:

```bash
# Linux/Mac
export HF_TOKEN=hf_xxx

# Windows PowerShell
$env:HF_TOKEN = "hf_xxx"
```

## Troubleshooting

**Lỗi `could not select device driver "nvidia"`**: chưa cài NVIDIA Container Toolkit hoặc GPU chưa bật trong Docker Desktop.

**Model chạy quá chậm (offload to CPU)**: VRAM không đủ. Đổi sang model nhỏ hơn (xem các option `Qwen2.5-VL-*-AWQ` hoặc `Qwen2-VL-2B-Instruct-AWQ`).

**Ảnh không tìm thấy**: kiểm tra TSV trong `docker/LMUData/` đã có path `/app/data/images/...` chưa. Chạy lại `python docker/prepare_docker_dataset.py` nếu cần.
