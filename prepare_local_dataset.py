"""
Chuyển đổi đường dẫn ảnh trong TSV gốc (Windows absolute path) sang
đường dẫn tuyệt đối trên máy Linux/WSL hiện tại để chạy trực tiếp qua Conda (không cần Docker).

Tự động nhận diện thư mục dự án hiện tại để xây dựng đường dẫn chính xác.
"""
import os
import re
import pandas as pd

# Thư mục gốc của dự án
WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))

# Thư mục chứa ảnh thực tế trên server
LOCAL_IMG_ROOT = os.path.join(WORKSPACE_ROOT, "dermnet-output", "dermnet-output", "images").replace("\\", "/")

# Thư mục chứa các file TSV của VLMEvalKit
SRC_DIR = os.path.join(WORKSPACE_ROOT, "VLMEvalKit", "LMUData")

FILES = ["DermNet_Test.tsv", "DermNet_Val_4k.tsv"]


def convert_path(p: str) -> str:
    """Chuyển đường dẫn ảnh Windows sang đường dẫn tuyệt đối trên Server Linux."""
    if not isinstance(p, str):
        return p
    # Chuẩn hoá dấu phân cách
    norm = p.replace("\\", "/")
    # Lấy phần tương đối sau segment '/images/'
    m = re.search(r"/images/(.+)$", norm)
    if m:
        rel = m.group(1)
    else:
        # Nếu đã là relative, lấy tên file hoặc phần đuôi
        rel = norm.split("/")[-1]
    
    # Kết hợp thành đường dẫn tuyệt đối trên máy hiện tại
    return f"{LOCAL_IMG_ROOT}/{rel}"


def main():
    print(f"Thư mục dự án: {WORKSPACE_ROOT}")
    print(f"Thư mục ảnh đích: {LOCAL_IMG_ROOT}")
    
    for fname in FILES:
        src = os.path.join(SRC_DIR, fname)
        if not os.path.exists(src):
            print(f"BỎ QUA (không tìm thấy): {src}")
            continue
        
        # Đọc dữ liệu TSV gốc
        df = pd.read_csv(src, sep="\t")
        col = "image_path" if "image_path" in df.columns else "image"
        
        # Áp dụng chuyển đổi đường dẫn
        df[col] = df[col].apply(convert_path)
        
        # Đảm bảo tên cột là image_path để VLMEvalKit đọc đúng
        if col == "image":
            df = df.rename(columns={"image": "image_path"})
            
        # Ghi đè trực tiếp lên tệp TSV trong VLMEvalKit/LMUData để chạy local
        df.to_csv(src, sep="\t", index=False)
        print(f"THÀNH CÔNG: Cập nhật {fname} ({len(df)} hàng) -> Ví dụ mẫu: {df['image_path'].iloc[0]}")


if __name__ == "__main__":
    main()
