# Hướng dẫn sử dụng Prompts cho AI Agent - Tư vấn Biểu diễn Biểu đồ

## Tổng quan

Bộ prompt này được thiết kế để giúp bạn nhận tư vấn từ AI Agent về cách biểu diễn tốt nhất các kết quả từ 4 kịch bản benchmark khác nhau trong dự án Vision Transformer Hashing.

---

## 📊 Bốn Kịch bản Benchmark

### **Kịch bản 1: So sánh Framework với Backbone Cố định**
- **File Prompt**: `prompt_scenario_001.md`
- **File Dữ liệu**: `assets/scenario_logs/scenario-001/summary.json`
- **Nội dung**: So sánh 4 framework (CSQ, IDHN, DPN, HashNet) trên ViT-B_32 & CIFAR-10 (32-bit)
- **Câu hỏi chính**: Framework nào hoạt động tốt nhất? Làm sao visualize sự hội tụ?
- **Kết quả tốt nhất**: IDHN (0.959 MAP)

### **Kịch bản 2: So sánh Backbone với Framework Cố định**
- **File Prompt**: `prompt_scenario_002.md`
- **File Dữ liệu**: `assets/scenario_logs/scenario-002/summary.json`
- **Nội dung**: So sánh 4 backbone (AlexNet, ResNet, ViT-B_16, ViT-B_32) với CSQ (32-bit)
- **Câu hỏi chính**: ViT backbones có vượt trội hơn CNN không? Visualize trade-off speed vs accuracy?
- **Kết quả tốt nhất**: ViT-B_16 (0.966 MAP)

### **Kịch bản 3: CLS Token vs All Tokens**
- **File Prompt**: `prompt_scenario_003.md`
- **File Dữ liệu**: `assets/scenario_logs/scenario-003/summary.json`
- **Nội dung**: So sánh 4 framework với 2 biến thể (all-tokens vs CLS-token variants) trên ViT-B_32
- **Câu hỏi chính**: Dùng CLS token có làm giảm performance không? Như thế nào?
- **Kết quả**: Sự khác biệt nhỏ (CLS token thường tốt hơn một chút hoặc bằng)

### **Kịch bản 5: Ảnh hưởng Chiều dài Hash Code**
- **File Prompt**: `prompt_scenario_005.md`
- **File Dữ liệu**: `assets/scenario_logs/scenario-005/summary.json`
- **Nội dung**: So sánh 3 chiều dài bit (16, 32, 64) với CSQ & ViT-B_32 trên CIFAR-10
- **Câu hỏi chính**: Tăng số bit có cải thiện MAP tuyến tính không? Diminishing returns ở đâu?
- **Kết quả**: 16-bit (0.960), 32-bit (0.953), 64-bit (0.957)

---

## 🚀 Cách Sử dụng

### **Bước 1: Chọn Kịch bản**
Quyết định bạn muốn tư vấn biểu đồ cho kịch bản nào (1, 2, 3 hay 5).

### **Bước 2: Đọc Prompt**
Mở file `prompt_scenario_XXX.md` tương ứng để hiểu bối cảnh và câu hỏi.

### **Bước 3: Chuẩn bị Dữ liệu**
File JSON log đã có sẵn tại:
```
assets/scenario_logs/scenario-001/summary.json
assets/scenario_logs/scenario-002/summary.json
assets/scenario_logs/scenario-003/summary.json
assets/scenario_logs/scenario-005/summary.json
```

### **Bước 4: Giao cho AI Agent**
Sao chép **toàn bộ nội dung prompt** (bao gồm context, questions, data summary) và paste vào AI Agent.

### **Bước 5: Thêm Dữ liệu JSON**
Nếu AI Agent cần chi tiết hơn, bạn có thể:
- Dán từng phần của file JSON (training losses, PR curves, final MAP scores)
- Hoặc upload file JSON trực tiếp nếu AI Agent hỗ trợ

---

## 📋 Dữ liệu Chính trong Mỗi File JSON

Mỗi file `summary.json` chứa:
- `"type": "train"` → Training loss tại mỗi epoch
- `"type": "pr"` → Precision-Recall curve values tại các checkpoint (epochs 30, 60, 90, 120, 150)
- `"type": "test"` → Test MAP score tại các checkpoint
- Mỗi model có `"best_map"` → best performance đạt được

**Ví dụ cấu trúc:**
```json
{
  "model_name_1": [
    {"type": "train", "epoch": 1, "loss": 0.684},
    {"type": "train", "epoch": 2, "loss": 0.155},
    ...
    {"type": "test", "epoch": 30, "map": 0.952, "best_map": 0.952},
    ...
  ],
  "model_name_2": [...],
  ...
}
```

---

## 💡 Gợi ý cho AI Agent

Khi tư vấn với AI Agent, bạn có thể nhấn mạnh:

1. **Mục tiêu**: Tạo biểu đồ cho bài báo nghiên cứu (publication-quality)
2. **Độc giả mục tiêu**: Nhà nghiên cứu trong lĩnh vực image hashing / computer vision
3. **Dữ liệu đặc trưng**: 
   - Convergence curves (training loss theo epoch)
   - Final performance comparison (bar chart MAP scores)
   - Precision-Recall curves tại các checkpoint chính
4. **Cân nhắc**:
   - Clarity (rõ ràng, dễ hiểu)
   - Consistency (phong cách thống nhất)
   - Color scheme (phù hợp in đen trắng & màu)
   - Legends & annotations (ghi chú rõ ràng)

---

## 📝 Ví dụ Câu hỏi bổ sung cho AI Agent

Sau khi AI Agent đưa ra gợi ý đầu tiên, bạn có thể hỏi thêm:

- "Bạn có thể vẽ cụ thể một ví dụ về cách visualization này không?"
- "Nên sử dụng loại biểu đồ nào khi có >4 models?"
- "Làm sao để highlight sự khác biệt giữa các methods khi chúng rất gần nhau?"
- "Code Python (matplotlib/seaborn/plotly) để vẽ biểu đồ này như thế nào?"

---

## 🎯 Kết quả Mong đợi

AI Agent sẽ cung cấp:
1. ✅ Khuyến nghị loại biểu đồ (line, bar, box, violin, v.v.)
2. ✅ Thứ tự/arrangement các models trong biểu đồ
3. ✅ Color scheme & styling recommendations
4. ✅ Gợi ý cách highlight key findings
5. ✅ Template hoặc code mẫu (nếu hỏi)

---

## 📌 Lưu ý

- **Scenario 4 không có**: Dữ liệu scenario 4 (so sánh multi-dataset) chưa được collect, chỉ có 4 scenarios
- **JSON files đã có**: Không cần tạo lại, chỉ cần sử dụng
- **Prompt có thể customize**: Bạn có thể sửa/thêm câu hỏi dựa trên nhu cầu cụ thể
- **AI Agent context**: Prompt được thiết kế để AI Agent hiểu bối cảnh research mà không cần giải thích thêm

---

## 📞 Liên hệ & Thay đổi

Nếu bạn cần:
- Thêm scenario khác
- Sửa prompts
- Tạo biểu đồ thực tế

Vui lòng cập nhật các files `prompt_scenario_XXX.md` tương ứng.
