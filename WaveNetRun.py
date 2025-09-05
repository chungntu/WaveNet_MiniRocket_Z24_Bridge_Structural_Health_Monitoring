# ============================================
# Chương trình huấn luyện WaveNet
# ============================================

import utils  # Thư viện utils.py: chứa find_latest_history_file, load_history_dict, plot_accuracy_from_history

from datasetManagement import datasetManagement  # Hàm chuẩn bị dữ liệu (cắt/chuẩn hóa/chia train-val-test)
from WaveNet import WavenetRun                   # Hàm huấn luyện/chạy mô hình WaveNet
import tensorflow as tf
from pathlib import Path

# --------------------------------------------
# Kiểm tra GPU khả dụng
# --------------------------------------------
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Lấy danh sách GPU và in thông tin (tên thiết bị)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print("GPU:", gpu.name)
else:
    print("No GPU devices found.")

# --------------------------------------------
# Xác định các lớp dùng để huấn luyện
#   - Ghi chú: lớp đầu tiên sẽ có nhãn 0, lớp thứ hai nhãn 1, ...
#   - Dải lớp hợp lệ: '01' đến '17' (tùy bộ dữ liệu)
# --------------------------------------------
# 17 lớp
classes = ['01', '03', '04', '05', '06','07','09','10','11','12','13','14','15','16','17']

# Dùng 5 lớp để huấn luyện
# classes = ['01', '03', '04', '05', '06']

# --------------------------------------------
# Chuẩn bị dữ liệu
#   - datasetManagement(classes, 65536) trả về:
#       X_train, X_val, X_test, y_train, y_val, y_test, width
#     trong đó width thường là chiều dài chuỗi (ví dụ 65536)
# --------------------------------------------
X_train, X_val, X_test, y_train, y_val, y_test, width = datasetManagement(classes, 65536)

# --------------------------------------------
# Biến 'model' đặt None để WavenetRun tự build mô hình mới.
# Nếu muốn load mô hình có sẵn thì bỏ comment phần "to load a model" phía dưới.
# --------------------------------------------
model = None

# --------------------------------------------
# to load a model (tuỳ chọn):
#   - Nếu muốn nạp mô hình đã huấn luyện trước, đặt đường dẫn chính xác
#   - compile=False để tránh lỗi nếu môi trường khác lúc compile ban đầu
# --------------------------------------------
# models_dir = Path(r"C:\Users\Dell Precision 7810\Documents\GitHub\WaveNet_MiniRocket_Z24_Bridge_Structural_Health_Monitoring")
# model_path = models_dir / "Wavenet8_1_9_65536_6.h5"
# model = tf.keras.models.load_model(str(model_path), compile=False)

# --------------------------------------------
# Siêu tham số huấn luyện (hyperparameters)
# --------------------------------------------
learning_rate = 0.0001           # Tốc độ học
filter = 8                       # Số filters (kênh) cho các Conv1D trong WaveNet
batchsize = 32                   # Kích thước batch (điều chỉnh theo VRAM)
epochs = 2                      # Số epoch huấn luyện
numberOfResidualsPerBlock = 9    # Số residual dilation trong 1 block: 2^0, 2^1, ..., 2^(N-1)
numberOfBlocks = 1               # Số block lặp lại (ví dụ =2: lặp lại chuỗi 2^0..2^9 hai lần)

# --------------------------------------------
# Gọi hàm huấn luyện WaveNet
#   - Nếu 'model' = None: WavenetRun sẽ tự khởi tạo model mới theo tham số
#   - Nếu 'model' đã load: WavenetRun sẽ tiếp tục train/đánh giá trên model đó
# --------------------------------------------
WavenetRun(
    model,
    filter,
    batchsize,
    epochs,
    learning_rate,
    numberOfResidualsPerBlock,
    numberOfBlocks,
    width,
    classes,
    X_train, X_val, X_test, y_train, y_val, y_test
)

# --------------------------------------------
# Vẽ Accuracy (train/val) trên cùng một hình & lưu PNG
#   - utils.find_latest_history_file("./History"): tìm file history *.txt mới nhất
#   - utils.load_history_dict(file): đọc dict {"accuracy":[...], "val_accuracy":[...], ...}
#   - utils.plot_accuracy_from_history(hist, save_path=...): vẽ & lưu hình
# --------------------------------------------
latest_hist_file = utils.find_latest_history_file("./History")
if latest_hist_file:
    print(f"[INFO] Đọc history từ: {latest_hist_file}")
    hist = utils.load_history_dict(latest_hist_file)
    # Lưu hình *_acc.png và đồng thời hiển thị trên màn hình (plt.show() nằm trong utils)
    utils.plot_accuracy_from_history(
        hist,
        save_path=latest_hist_file.replace(".txt", "_acc.png")
    )
