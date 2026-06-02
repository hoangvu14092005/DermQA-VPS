# Hướng dẫn đưa lên Git và Triển khai trên Linux WSL

Tài liệu này hướng dẫn chi tiết cách cấu hình Git cho dự án `Dermnet-QA`, dọn dẹp các tệp tin không mong muốn, cài đặt môi trường trên máy Linux WSL mới và chạy đánh giá mô hình.

---

## 1. Những việc đã được thiết lập tự động

1. **Dọn dẹp thư mục:**
   - Đã xóa thư mục trống thừa `-Force/`.
   - Đã xóa thư mục ẩn `VLMEvalKit/.git/` để mã nguồn tùy chỉnh của `VLMEvalKit` được đẩy lên cùng dự án chính (thành một monorepo).
2. **Cấu hình Git & .gitignore:**
   - Đã tạo tệp `.gitignore` để bỏ qua các file dữ liệu nặng (`dermnet-output/`, `docker/hf_cache/`, `docker/outputs/`) và khóa bảo mật nhạy cảm (`mykey.pem`).
   - Đã khởi tạo Git repository (`git init`).
   - Đã cấu hình remote `origin` trỏ về `https://github.com/hoangvu14092005/DermQA-VPS`.

---

## 2. Hướng dẫn các bước tiếp theo

### Bước 1: Commit và Push code lên GitHub (Từ máy Windows hiện tại)
Hãy mở terminal (PowerShell hoặc CMD) tại thư mục dự án trên máy Windows và chạy lần lượt các lệnh sau:

```powershell
# 1. Thêm toàn bộ các tệp hợp lệ vào Git tracking
git add .

# 2. Tạo commit đầu tiên
git commit -m "Initial commit - DermNet VLM Eval Docker Setup"

# 3. Đổi tên nhánh mặc định thành main (nếu chưa có)
git branch -M main

# 4. Đẩy mã nguồn lên GitHub
git push -u origin main
```

---

### Bước 2: Cài đặt trên máy Linux WSL mới (Giả sử máy chưa cài gì)
Khi bạn đã chuyển sang máy Linux WSL mới, hãy mở terminal WSL (Ubuntu) và chạy lần lượt các lệnh sau:

#### 1. Cập nhật hệ thống và cài đặt Python, Git
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3 python3-pip python3-venv curl
```

#### 2. Cấu hình Docker
*Khuyên dùng:* Cài đặt **Docker Desktop** trên Windows và kích hoạt tính năng **WSL Integration** (kết nối với Ubuntu) trong phần cài đặt của Docker Desktop. Cách này sẽ tự động tích hợp GPU NVIDIA và Docker vào WSL mà không cần cấu hình phức tạp.

*Nếu bạn chỉ cài Docker Engine độc lập trực tiếp trong WSL:*
```bash
# Cài đặt Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

#### 3. Cài đặt NVIDIA Container Toolkit (Bắt buộc để Docker dùng GPU)
*(Chỉ cần thiết nếu bạn cài Docker Engine độc lập trong WSL; nếu dùng Docker Desktop kết nối WSL thì có thể bỏ qua bước này)*
```bash
# Cài đặt kho lưu trữ
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Khởi động lại Docker
sudo nvidia-container-toolkit-config --mode=docker
sudo systemctl restart docker
```

#### 4. Cài đặt thư viện Pandas trên WSL Host
```bash
pip3 install pandas
```

---

### Bước 3: Chạy dự án trên máy WSL mới

Sau khi máy mới đã được cài đặt đầy đủ:

1. **Tải code từ Git:**
   ```bash
   git clone https://github.com/hoangvu14092005/DermQA-VPS.git
   cd DermQA-VPS
   ```
2. **Copy Dataset vào dự án:**
   * Hãy đưa thư mục ảnh `dermnet-output/` vào thư mục của dự án trên WSL tại đường dẫn: `./dermnet-output/dermnet-output/images`.
3. **Chuyển đổi đường dẫn Dataset phù hợp với Docker:**
   ```bash
   python3 docker/prepare_docker_dataset.py
   ```
4. **Build và Khởi chạy Docker:**
   ```bash
   # Thiết lập HF token nếu cần
   export HF_TOKEN="your_huggingface_token"
   
   # Build & Chạy
   docker compose build
   docker compose up
   ```

---

## 3. Cấu hình Model & Yêu cầu VRAM (Tips)

Bộ thư viện **VLMEvalKit** hỗ trợ rất nhiều mô hình. Dưới đây là cách bạn cấu hình và lựa chọn mô hình phù hợp với cấu hình phần cứng:

### 1. Cách thay đổi mô hình chạy
Mở tệp `docker-compose.yml`, tại phần `environment:`, bạn hãy sửa giá trị của biến `MODEL`:
```yaml
environment:
  MODEL: "Qwen2.5-VL-3B-Instruct-AWQ" # Thay tên model tại đây
```

### 2. Các mô hình khuyên dùng & Yêu cầu phần cứng (Local GPU)
*   **`Qwen2.5-VL-3B-Instruct-AWQ`** (Mặc định):
    *   *Yêu cầu VRAM:* **6GB - 8GB VRAM** (Phù hợp cho card RTX 3060, RTX 4050, RTX 4060).
    *   *Đặc điểm:* Bản lượng tử hóa nhẹ, độ chính xác khá tốt và thời gian suy luận nhanh.
*   **`Qwen2.5-VL-7B-Instruct-AWQ`**:
    *   *Yêu cầu VRAM:* **12GB - 16GB VRAM** (Phù hợp cho card RTX 3060 12GB, RTX 4070, RTX 4080).
    *   *Đặc điểm:* Độ chính xác tốt hơn bản 3B, nhận diện tốt các đặc trưng ảnh y tế phức tạp.
*   **`Qwen2.5-VL-3B-Instruct`** / **`Qwen2.5-VL-7B-Instruct`**:
    *   Các bản không lượng tử (FP16), yêu cầu VRAM cao gấp 1.5 - 2 lần bản AWQ tương ứng.

> [!WARNING]
> Các mô hình lớn hơn như **`Qwen2.5-VL-32B-Instruct`** hoặc **`72B-Instruct`** yêu cầu phần cứng chuyên dụng (tối thiểu **40GB - 80GB VRAM** như card A100/H100) để có thể chạy offline.

### 3. Sử dụng API để giảm tải phần cứng (Nếu GPU yếu)
Nếu máy của bạn không có GPU mạnh, bạn có thể chuyển sang sử dụng mô hình qua API của các nhà cung cấp như OpenAI hay Google Gemini. Việc này yêu cầu bạn cấu hình API Key tương ứng trong biến môi trường:
*   **Google Gemini (Khuyên dùng vì xử lý tiếng Việt rất tốt):** Cấu hình `MODEL` thành `GeminiFlash2-0` hoặc `GeminiPro1-5` trong `docker-compose.yml`.
*   **OpenAI GPT:** Cấu hình `MODEL` thành `GPT4o` hoặc `GPT4o_MINI`.
