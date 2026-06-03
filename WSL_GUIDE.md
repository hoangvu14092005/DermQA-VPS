# Quy trình Triển khai và Chạy Đánh giá qua Miniconda (Conda)

Tài liệu này hướng dẫn chi tiết quy trình chạy **từng bước lần lượt** sử dụng **Miniconda (Conda)** trực tiếp trên Server GPU dùng chung (như `ai-train-a40-2`).

> [!NOTE]
> Cách chạy qua Conda giúp cô lập môi trường hoàn hảo, không cần quyền quản trị (`sudo`), không cần cài đặt Docker hệ thống, và tối ưu hiệu năng phần cứng tốt nhất.

---

## BƯỚC 1: Đẩy toàn bộ dự án lên Git (Từ máy Windows của bạn)

Mở Terminal (PowerShell hoặc CMD) tại thư mục dự án `Dermnet-QA` trên máy Windows và chạy lần lượt các lệnh sau:

```powershell
# 1. Thêm toàn bộ mã nguồn, tệp tin cấu hình và dữ liệu vào Git
git add .

# 2. Tạo commit đầu tiên
git commit -m "Initial commit - Conda setup with full dataset"

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

## BƯỚC 5: Đồng bộ đường dẫn Dataset trên Server (Bắt buộc)

Do tệp tin TSV bạn đẩy lên Git có chứa đường dẫn ảnh định dạng Windows (dùng dấu gạch chéo ngược `\`), hệ thống Linux trên Server sẽ không đọc được (báo lỗi SKIP không tìm thấy ảnh). Bạn bắt buộc phải chạy script sau để tự động chuyển đổi sang đường dẫn Linux:

```bash
# 1. Di chuyển về thư mục gốc của dự án
cd ~/tungns/DermQA-VPS

# 2. Chạy script đồng bộ đường dẫn ảnh (script sẽ tự động quét và sửa toàn bộ các file .tsv trong LMUData)
python3 prepare_local_dataset.py
```
*(Nếu thành công, terminal sẽ thông báo `THÀNH CÔNG: Cập nhật ...` kèm đường dẫn ví dụ trên Linux).*

---

## BƯỚC 6: Khởi chạy đánh giá mô hình

Đảm bảo bạn vẫn đang ở trong môi trường ảo `(dermnet)`. Vì các bộ dữ liệu của bạn là Custom (tự định nghĩa), thư viện VLMEvalKit không hỗ trợ tính điểm tự động (sẽ báo lỗi `NotImplementedError`). Do đó, bạn cần chạy ở chế độ **chỉ dự đoán (Inference Only)** bằng cách thêm tham số `--mode infer`:

```bash
# 1. Di chuyển vào thư mục VLMEvalKit
cd ~/tungns/DermQA-VPS/VLMEvalKit

# 2. Chạy lệnh đánh giá mô hình Qwen2.5-VL-3B-Instruct-AWQ (mặc định)
# Lưu ý bắt buộc phải có `--mode infer` đối với custom dataset để tránh lỗi
LMUData=~/tungns/DermQA-VPS/VLMEvalKit/LMUData CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 run.py --data DermNet_Val_4k --model Qwen2.5-VL-3B-Instruct-AWQ --work-dir ../outputs --verbose --reuse --mode infer
```
*Sau khi chạy xong, kết quả dạng file `.xlsx` chứa dự đoán của mô hình sẽ tự động xuất hiện tại thư mục `~/tungns/DermQA-VPS/outputs/`.*

---

## BƯỚC 7: Cấu hình thay đổi Model & Yêu cầu VRAM (Mẹo thêm)

Nếu bạn muốn thay đổi mô hình khác để đánh giá, tại lệnh chạy ở **BƯỚC 6**, hãy thay đổi tham số `--model <tên_model>`. Dưới đây là các mô hình khuyên dùng phù hợp với cấu hình GPU của Server (A40 với 48GB VRAM):

*   **`Qwen2.5-VL-7B-Instruct-AWQ`** (Khuyên dùng cho độ chính xác cao & tốc độ nhanh):
    *   *Lệnh chạy:*
        ```bash
        LMUData=~/tungns/DermQA-VPS/VLMEvalKit/LMUData CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 run.py --data DermNet_Val_4k --model Qwen2.5-VL-7B-Instruct-AWQ --work-dir ../outputs --verbose --reuse --mode infer
        ```
*   **`Qwen2.5-VL-7B-Instruct`** (Bản FP16 đầy đủ cho chất lượng tốt nhất):
    *   *Lệnh chạy:*
        ```bash
        LMUData=~/tungns/DermQA-VPS/VLMEvalKit/LMUData CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 run.py --data DermNet_Val_4k --model Qwen2.5-VL-7B-Instruct --work-dir ../outputs --verbose --reuse --mode infer
        ```
*   **`Qwen2-VL-2B-Instruct`** (Mô hình Qwen2-VL 2B):
    *   *Lệnh chạy:*
        ```bash
        LMUData=~/tungns/DermQA-VPS/VLMEvalKit/LMUData CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 run.py --data DermNet_Val_4k --model Qwen2-VL-2B-Instruct --work-dir ../outputs --verbose --reuse --mode infer
        ```
