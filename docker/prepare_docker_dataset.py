"""
Chuyển đổi đường dẫn ảnh trong TSV gốc (Windows absolute path) sang
đường dẫn cố định trong container (/app/data/images/...).

Giữ nguyên TSV gốc trong VLMEvalKit/LMUData (dùng cho chạy local Windows).
Tạo TSV mới trong docker/LMUData (dùng khi chạy trong container).
"""
import os
import re

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "VLMEvalKit", "LMUData")
OUT_DIR = os.path.join(os.path.dirname(__file__), "LMUData")
# Đường dẫn ảnh bên trong container (sẽ mount thư mục images vào đây)
CONTAINER_IMG_ROOT = "/app/data/images"

FILES = ["DermNet_Test.tsv", "DermNet_Val_4k.tsv"]


def convert_path(p: str) -> str:
    """Chuyển 1 đường dẫn ảnh thành path container, lấy phần sau 'images'."""
    if not isinstance(p, str):
        return p
    # Chuẩn hoá dấu phân cách
    norm = p.replace("\\", "/")
    # Lấy phần sau segment '/images/' cuối cùng
    m = re.search(r"/images/(.+)$", norm)
    if m:
        rel = m.group(1)
    else:
        # Nếu đã là relative, giữ nguyên phần đuôi
        rel = norm.lstrip("/")
    return f"{CONTAINER_IMG_ROOT}/{rel}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    import pandas as pd

    for fname in FILES:
        src = os.path.join(SRC_DIR, fname)
        if not os.path.exists(src):
            print(f"SKIP (not found): {src}")
            continue
        df = pd.read_csv(src, sep="\t")
        col = "image_path" if "image_path" in df.columns else "image"
        df[col] = df[col].apply(convert_path)
        # Đảm bảo tên cột là image_path để VLMEvalKit đọc theo file path
        if col == "image":
            df = df.rename(columns={"image": "image_path"})
        out = os.path.join(OUT_DIR, fname)
        df.to_csv(out, sep="\t", index=False)
        print(f"OK: {out}  ({len(df)} rows)  sample -> {df['image_path'].iloc[0]}")


if __name__ == "__main__":
    main()
