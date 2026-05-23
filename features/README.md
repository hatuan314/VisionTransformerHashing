# Features

Các script dùng để build database index và tìm kiếm ảnh sau khi đã train model.

---

## Cài đặt môi trường Python

> Chạy các lệnh sau từ thư mục gốc của project (`VisionTransformerHashing/`).

### 1. Tạo virtual environment

```bash
python -m venv venv
```

### 2. Kích hoạt môi trường

```bash
# Linux / macOS
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

> **Lưu ý GPU:** Lệnh trên cài PyTorch CPU-only. Nếu muốn dùng CUDA, cài đúng phiên bản theo hướng dẫn tại [pytorch.org](https://pytorch.org/get-started/locally/), ví dụ:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> pip install numpy scipy matplotlib Pillow tqdm ml-collections
> ```

### 4. Kiểm tra cài đặt

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## Tổng quan

```
features/
├── build_database.py       # Build database hash index từ model đã train
└── search_top10_uploaded.py # Tìm kiếm top-10 ảnh tương tự
```

---

## build_database.py

Tạo database hash index từ một model trong `train-models/`. Script tự động parse tên file model để lấy cấu hình (dataset, backbone, số bit) mà không cần chỉnh tay.

### Cách dùng

**Liệt kê các model có sẵn:**
```bash
python features/build_database.py --list
```

**Chọn model tương tác (không truyền tham số):**
```bash
python features/build_database.py
```

**Chỉ định model cụ thể:**
```bash
python features/build_database.py --model cifar10_CSQ_ViT-B_32_Bit64-BestModel.pth
```

### Định dạng tên model

Script nhận diện file model theo pattern:
```
{dataset}_{method}_{backbone}_Bit{bits}-{BestModel|IntermediateModel}.{pt|pth}
```

Ví dụ:
- `cifar10_CSQ_ViT-B_32_Bit64-BestModel.pth`
- `cifar10_DSH_ViT-B_32_Bit32-BestModel.pt`

### Output

Database index được lưu vào thư mục con trong `database_index/`:
```
database_index/{dataset}_{method}_{backbone}_Bit{bits}/
├── db_codes.pt     # Hash codes [N, bits]
├── db_labels.pt    # Labels [N, n_class]
└── db_indices.pt   # Global indices [N]
```

---

## search_top10_uploaded.py

Nhận một ảnh query, hash bằng model đã train, rồi tìm top-10 ảnh tương tự nhất trong database index bằng khoảng cách Hamming.

### Yêu cầu trước khi chạy

1. **Đã build database index** — chạy `build_database.py` trước (xem hướng dẫn bên trên).
2. **Ảnh query** — chuẩn bị file ảnh muốn tìm kiếm (ví dụ: `query.jpg`).

### Chỉnh cấu hình trong script

Mở `features/search_top10_uploaded.py` và sửa 3 dòng trong hàm `main()`:

```python
bit = 64                                            # số bit hash (phải khớp với model)
model_path = "cifar10_CSQ_ViT-B_32_Bit64-BestModel.pth"  # tên file model trong Checkpoints_Results/
query_image_path = "query.jpg"                      # đường dẫn ảnh query
```

Và đường dẫn database index (mặc định dùng thư mục `database_index/`):

```python
db_codes  = torch.load("database_index/db_codes.pt")
db_labels = torch.load("database_index/db_labels.pt")
db_indices= torch.load("database_index/db_indices.pt")
```

> **Lưu ý:** Thư mục `database_index/` phải khớp với output của `build_database.py`.  
> Nếu bạn build với `--model cifar10_CSQ_ViT-B_32_Bit64-BestModel.pth` thì index nằm ở  
> `database_index/cifar10_CSQ_ViT-B_32_Bit64/` — hãy sửa đường dẫn cho đúng.

### Chạy script

> Chạy từ thư mục **gốc** của project (`VisionTransformerHashing/`), không phải bên trong `features/`.

```bash
# Kích hoạt venv trước
source venv/bin/activate

# Chạy
python features/search_top10_uploaded.py
```

### Output

- **Terminal:** in thông tin hash code query, top-10 nhãn, khoảng cách Hamming từng kết quả.
- **File ảnh:** `top10_uploaded_result.png` — lưới ảnh gồm ảnh query (trái) + 10 ảnh gần nhất (kèm nhãn và khoảng cách Hamming).

### Ví dụ output terminal

```
Loading model...
Loading precomputed database index...
db_codes: torch.Size([50000, 64])
db_labels: torch.Size([50000, 10])
db_indices: torch.Size([50000])
Loading dataset again for image display...
Loading uploaded query image...

================ QUERY INFO ================
Uploaded image path: query.jpg
Query hash code (0/1): [1, 0, 1, ...]

Computing Hamming distance...

================ TOP 10 RESULTS ================
Top 10 labels (name): ['cat', 'cat', 'dog', ...]
Top 10 distances: [3.0, 4.0, 5.0, ...]

Saved result to top10_uploaded_result.png
```
