# Hệ Thống Dự Đoán Bệnh Tim — D-DOA + SVM

Mô hình: D-DOA v4 (Deep Drizzle Optimization Algorithm) + SVM-RBF  
Dữ liệu: UCI Heart Disease Dataset (Cleveland, Hungarian, VA, Switzerland)  
Giao diện: Streamlit Web App  
Link demo: https://heart-disease-prediction-4e8t7nl7drkzgm7xur5eqd.streamlit.app/

---

## Giới thiệu

Đây là hệ thống hỗ trợ sàng lọc nguy cơ bệnh tim, được xây dựng trong khuôn khổ đồ án chuyên ngành. Hệ thống áp dụng thuật toán Drizzle Optimization Algorithm phiên bản Deep (D-DOA v4) để tối ưu đồng thời siêu tham số C, gamma của mô hình SVM-RBF và trọng số của từng đặc trưng lâm sàng, thay vì dùng Grid Search thông thường.

Lưu ý: Đây là công cụ hỗ trợ nghiên cứu, không thay thế chẩn đoán y khoa. Kết quả chỉ mang tính tham khảo. Người dùng cần gặp bác sĩ chuyên khoa tim mạch để được đánh giá chính xác.

---

## Hiệu suất mô hình

Kết quả chạy với n_iter = 2000, SEED = 42:

| Chỉ số | Giá trị |
|---|---|
| Test Accuracy | 84.78% |
| AUC-ROC | 0.9165 |
| F1-Score | 0.8667 |
| MCC | 0.6913 |
| CV AUC-ROC (5-fold) | 89.79% |
| Tham số tối ưu | C = 1.0000, gamma = 0.073940 |

---

## Yêu cầu cài đặt

```
Python >= 3.9
streamlit
numpy
pandas
scikit-learn
imbalanced-learn
matplotlib
```

Cài đặt:

```bash
pip install streamlit numpy pandas scikit-learn imbalanced-learn matplotlib
```

---

## Dữ liệu

Ứng dụng dùng 4 file từ UCI Heart Disease:

```
processed.cleveland.data
processed.hungarian.data
processed.va.data
processed.switzerland.data
```

Cách cung cấp dữ liệu:
1. Tải về tại: https://archive.ics.uci.edu/dataset/45/heart+disease
2. Giải nén vào thư mục `~/Downloads/heart+disease/`
3. Nếu không tìm thấy file cục bộ, ứng dụng tự động tải từ UCI khi khởi động.

---

## Hướng dẫn sử dụng

### Bước 1 — Khởi động

Chạy lệnh sau trong terminal:

```bash
streamlit run giaodienchinh1.py
```

Hoặc truy cập trực tiếp qua link Streamlit Cloud ở trên. Khi mở lần đầu, giao diện hiển thị panel tải mô hình.

---

### Bước 2 — Tải mô hình

Có hai lựa chọn:

**Lựa chọn 1 — Dùng kết quả đã tối ưu (khuyến dùng)**

Kết quả từ quá trình chạy DOA 2000 vòng lặp (khoảng 32,5 phút) đã được lưu sẵn trong chương trình. Bấm nút "Tải kết quả đã tối ưu", hệ thống sẽ tái tạo mô hình theo đúng pipeline gốc và sẵn sàng dự đoán sau vài giây.

Pipeline tái tạo:
```
X_raw x best_weights
  -> train_test_split 80/20 (stratify, seed=42)
  -> SMOTE chi tren tap train (k_neighbors=3)
  -> StandardScaler (fit tren train, transform test)
  -> SVM-RBF (C=1.0, gamma=0.073940)
  -> Danh gia tren tap test thuc
```

**Lựa chọn 2 — Train lại**

Người dùng tự điều chỉnh tham số và chạy lại DOA:

| Tham số | Mặc định | Phạm vi |
|---|---|---|
| n_droplets | 30 | 10 – 50 |
| n_iter | 60 | 20 – 200 |
| CV Folds | 5 | 3 – 10 |

Lưu ý: Train lại với n_iter nhỏ sẽ cho kết quả khác và thường thấp hơn mô hình gốc.

---

### Bước 3 — Phân tích đơn lẻ

Sau khi tải mô hình xong, giao diện chuyển sang tab "Phân tích Đơn lẻ" với bố cục hai cột.

**Cột trái — Nhập thông tin bệnh nhân**

| Trường | Mô tả | Đơn vị / Lựa chọn |
|---|---|---|
| Họ và Tên | Tên bệnh nhân | Văn bản |
| Tuổi | Độ tuổi | 20 – 100 |
| Giới tính | Giới tính | Nam (1), Nữ (0) |
| Chiều cao | Chiều cao cơ thể | cm (100 – 250) |
| Cân nặng | Cân nặng cơ thể | kg (30 – 200) |
| Loại đau ngực | Phân loại cơn đau ngực | Typical Angina, Atypical Angina, Non-anginal, Asymptomatic |
| Huyết áp | Huyết áp lúc nghỉ | mmHg (80 – 220) |
| Cholesterol | Cholesterol huyết thanh | mg/dl (100 – 600) |
| Đường huyết đói > 120 | Có vượt ngưỡng không | Có (1), Không (0) |
| Kết quả ECG | Điện tâm đồ lúc nghỉ | Normal, ST-T Abnormality, LV Hypertrophy |
| Nhịp tim tối đa | Nhịp tim đạt được khi gắng sức | bpm (60 – 220) |
| Đau ngực khi vận động | Exercise-induced angina | Có (1), Không (0) |
| ST Depression | Oldpeak so với lúc nghỉ | 0.0 – 10.0 |
| Độ dốc ST | Hướng đoạn ST khi gắng sức | Upsloping, Flat, Downsloping |
| Mạch máu chính bị hẹp | Số mạch hẹp qua fluoroscopy | 0, 1, 2, 3 |
| Thalassemia | Kết quả xét nghiệm máu | Normal (3), Fixed Defect (6), Reversible Defect (7) |

Sau khi điền xong, bấm nút "Phân tích".

**Cột phải — Kết quả dự đoán**

Kết quả hiển thị gồm:

- Banner xanh (khoẻ mạnh) hoặc đỏ (có nguy cơ bệnh tim), kèm tên bệnh nhân và các chỉ số nhanh (tuổi, giới tính, BMI, huyết áp).
- Hình minh hoạ cơ thể 2D thay đổi màu sắc theo kết quả và chỉ số BMI.
- Các chỉ số tim mạch chi tiết: huyết áp, cholesterol, nhịp tim, đường huyết, kết quả ECG, số mạch bị hẹp. Mỗi chỉ số được tô màu xanh/vàng/đỏ theo ngưỡng y tế chuẩn.
- Xác suất dự đoán: phần trăm khoẻ mạnh và phần trăm nguy cơ bệnh tim.
- Khuyến nghị y tế dựa trên kết quả, tham chiếu từ AHA, ACC, WHO, CDC. Có cảnh báo riêng nếu huyết áp >= 140 mmHg, cholesterol >= 240 mg/dl, hoặc BMI >= 25.

---

### Bước 4 — Phân tích hàng loạt (Batch CSV)

Tab "Phân tích Hàng loạt CSV" cho phép dự đoán nhiều bệnh nhân cùng lúc.

**Định dạng file CSV đầu vào:**

File phải có các cột sau (đúng tên, đúng thứ tự):

```
patient_name, age, sex, height, weight, cp, trestbps, chol, fbs,
restecg, thalach, exang, oldpeak, slope, ca, thal
```

Ví dụ một dòng dữ liệu:

```
Nguyen Van A, 55, 1, 170, 72, 4, 140, 250, 1, 2, 130, 1, 2.3, 2, 1, 7
```

Quy trình sau khi upload:

1. Hệ thống kiểm tra và báo lỗi nếu file thiếu cột hoặc có giá trị ngoài phạm vi hợp lệ.
2. Dự đoán lần lượt từng bệnh nhân.
3. Hiển thị bảng tổng kết: tổng số bệnh nhân, số khoẻ mạnh, số nguy cơ, tỉ lệ nguy cơ.
4. Hiển thị 3 biểu đồ: phân bố kết quả (pie chart), phân phối xác suất bệnh tim (histogram), phân bố tuổi theo kết quả.
5. Bảng chi tiết có thể lọc theo kết quả và sắp xếp theo nhiều tiêu chí.
6. Xuất kết quả ra file CSV (toàn bộ hoặc theo bộ lọc hiện tại).

---

### Bước 5 — Lịch sử dự đoán

Mỗi lần bấm "Phân tích" trong tab đơn lẻ, kết quả được lưu vào lịch sử trong phiên làm việc. Lịch sử có thể xuất ra CSV bằng nút "Tải CSV". Bấm "Reset" để xoá lịch sử và bắt đầu phiên mới.

---

## Cấu trúc file

```
giaodienchinh1.py                        # File chính chạy Streamlit
Heart1_fixed_v4_smote_fixed.ipynb        # Notebook huấn luyện DOA gốc (n_iter=2000)
DOA_vs_GridSearch_v7_fixed.ipynb         # Notebook so sánh DOA vs Grid Search
README.md                                # File hướng dẫn này
```

---

## Mô tả thuật toán D-DOA v4

D-DOA mô phỏng hành vi của các giọt mưa tìm đường chảy xuống, kết hợp 5 cơ chế tối ưu:

| Cơ chế | Mô tả |
|---|---|
| Raindrop (PSO) | Cập nhật vị trí theo quán tính và vị trí tốt nhất cá nhân |
| Global Attract | Kéo giọt về phía vị trí tốt nhất toàn cục |
| Neighbourhood Sharing | Chia sẻ thông tin với các giọt lân cận gần nhất |
| Condensation Perturbation | Nhiễu Gaussian quanh trung tâm elite để thoát cực trị cục bộ |
| Levy Surge | Bước nhảy Levy Flight khi giọt bị đình trệ quá lâu |

Không gian tìm kiếm gồm 15 chiều: 13 trọng số đặc trưng + log(C) + log(gamma). Hàm fitness là 1 - CV AUC-ROC trung bình (5-fold StratifiedKFold).

---

## Tác giả

Phạm Minh Quân — MSSV 2045230079  
Trường Đại học Công Thương TP.HCM (HUIT)  
Ngành Công nghệ Thông tin  
Đồ án chuyên ngành — Nhóm 11
