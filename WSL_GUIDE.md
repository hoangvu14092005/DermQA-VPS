# Quy trình Triển khai và Chạy Đánh giá qua Miniconda (Conda)

Tài liệu này hướng dẫn chi tiết quy trình chạy **từng bước lần lượt** sử dụng **Miniconda (Conda)** trực tiếp trên Server GPU dùng chung (như `ai-train-a40-2`). 

> [!NOTE]
> Cách chạy qua Conda giúp cô lập môi trường hoàn hảo, không cần quyền quản trị (`sudo`), không cần cài đặt Docker hệ thống, và tối ưu hiệu năng phần cứng tốt nhất.

---

## BƯỚC 1: Đẩy toàn bộ dự án lên Git (Từ máy Windows của bạn)

Mở Terminal (PowerShell hoặc CMD) tại thư mục dự án `Dermnet-QA` trên máy Windows và chạy lần lượt các lệnh sau:

```powershell
# 1. Thêm toàn bộ mã nguồn, tệp tin cấu hình và ảnh dataset vào Git
git add .

# 2. Tạo commit đầu tiên
git commit -m "Initial commit - Conda setup with dataset"

# 3. Đặt tên nhánh là main
git branch -M main

# 4. Đẩy mã nguồn lên GitHub
git push -u origin main
```

---

## BƯỚC 2: Kết nối SSH và Tải mã nguồn về Server

1. Kết nối SSH vào server của bạn (User: `tp`, Host: `118.138.238.214`).
2. Chạy lần lượt các lệnh sau trên terminal của server để tải code về thư mục cá nhân:

```bash
# 1. Di chuyển vào thư mục làm việc cá nhân của bạn
cd ~/tungns

# 2. Tải dự án từ GitHub về thư mục này
git clone https://github.com/hoangvu14092005/DermQA-VPS.git

# 3. Di chuyển vào thư mục dự án vừa tải về
cd DermQA-VPS
```

---

## BƯỚC 3: Thiết lập môi trường ảo Conda

Vì máy chủ đã được cài đặt sẵn Conda (hiển thị chữ `(base)` ở đầu dòng lệnh), bạn chạy các lệnh sau để khởi tạo môi trường riêng:

```bash
# 1. Tạo môi trường ảo mới tên là dermnet (dùng Python 3.10)
conda create -n dermnet python=3.10 -y

# 2. Kích hoạt môi trường ảo lên (tiền tố terminal sẽ chuyển sang (dermnet))
conda activate dermnet

# 3. Cài đặt các thư viện cơ bản cho việc xử lý file dữ liệu
pip install pandas openpyxl
```

---

## BƯỚC 4: Cài đặt các thư viện cho VLMEvalKit

Cài đặt các thư viện tính năng của bộ kit đánh giá trực tiếp trong môi trường ảo của bạn:

```bash
# 1. Di chuyển vào thư mục VLMEvalKit
cd ~/tungns/DermQA-VPS/VLMEvalKit

# 2. Cài đặt các thư viện phụ thuộc (sử dụng file không chứa polygon3 để tránh lỗi biên dịch C++)
pip install -r requirements_no_polygon.txt

# 3. Cài đặt VLMEvalKit ở chế độ editable
pip install -e . --no-deps

# 4. Cài đặt phiên bản sympy tương thích để tránh lỗi tính toán
pip install sympy==1.13.1
```

---

## BƯỚC 5: Đồng bộ đường dẫn Dataset trên Server

Vì các tệp dữ liệu TSV gốc lưu đường dẫn ảnh theo hệ điều hành Windows, ta cần chạy script Python để tự động nhận diện thư mục làm việc hiện tại và cập nhật đường dẫn ảnh phù hợp với Linux:

```bash
# 1. Di chuyển về thư mục gốc của dự án
cd ~/tungns/DermQA-VPS

# 2. Chạy script đồng bộ đường dẫn ảnh
python3 prepare_local_dataset.py
```
*(Nếu thành công, terminal sẽ thông báo `THÀNH CÔNG: Cập nhật ...` kèm đường dẫn ví dụ trên Linux).*

---

## BƯỚC 6: Khởi chạy đánh giá mô hình

Đảm bảo bạn vẫn đang ở trong môi trường ảo `(dermnet)`, chạy lệnh sau để thực thi quá trình đánh giá:

```bash
# 1. Di chuyển vào thư mục VLMEvalKit
cd ~/tungns/DermQA-VPS/VLMEvalKit

# 2. Chạy lệnh đánh giá mô hình Qwen2.5-VL-3B-Instruct-AWQ (mặc định)
python run.py --data DermNet_Test DermNet_Val_4k --model Qwen2.5-VL-3B-Instruct-AWQ --work-dir ../outputs --verbose
```
*Sau khi chạy xong, kết quả dạng file `.xlsx` sẽ tự động xuất hiện tại thư mục `~/tungns/DermQA-VPS/outputs/`.*

---

## BƯỚC 7: Cấu hình thay đổi Model & Yêu cầu VRAM (Mẹo thêm)

Nếu bạn muốn thay đổi mô hình khác để đánh giá, tại lệnh chạy ở **BƯỚC 6**, hãy thay đổi tham số `--model <tên_model>`. Dưới đây là các mô hình khuyên dùng phù hợp với cấu hình GPU:

*   **`Qwen2.5-VL-3B-Instruct-AWQ`** (Mặc định - Khuyên dùng cho GPU yếu):
    *   *Yêu cầu VRAM:* **6GB - 8GB VRAM** (RTX 3060, RTX 4060).
*   **`Qwen2.5-VL-7B-Instruct-AWQ`** (Cho kết quả chính xác hơn):
    *   *Yêu cầu VRAM:* **12GB - 16GB VRAM** (RTX 3060 12GB, RTX 4070, A40).
*   **`Qwen2.5-VL-32B-Instruct`** hoặc **`72B-Instruct`**:
    *   *Yêu cầu VRAM:* **40GB - 80GB VRAM** (Cực kỳ nặng, chỉ chạy được trên GPU chuyên dụng hiệu năng cao).
*   **Sử dụng API (Không tốn VRAM):**
    *   Thay đổi `--model` thành `GeminiFlash2-0` hoặc `GPT4o_MINI` (Yêu cầu set API Key trong biến môi trường bằng lệnh `export GEMINI_API_KEY="key_của_bạn"` trước khi chạy).
