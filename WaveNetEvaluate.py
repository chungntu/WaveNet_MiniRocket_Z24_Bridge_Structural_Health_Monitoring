# ============================================
# ĐÁNH GIÁ MÔ HÌNH WAVENET TRÊN TẬP TEST
# (Giữ nguyên logic; bổ sung phần duyệt file .h5 bằng hộp thoại + comment đầy đủ)
# ============================================

from datasetManagement import datasetManagement  # Hàm quản lý dataset (cắt/chia train-val-test)
import utils                                     # Thư viện tiện ích: evaluate_NN_models, print_table, draw_confusion_matrix
import tensorflow as tf
from pathlib import Path
import numpy as np

# --------------------------------------------
# 1. Khai báo các lớp dùng để huấn luyện/đánh giá
#    - Thứ tự trong list: phần tử đầu tiên được gán nhãn 0, tiếp theo là 1, ...
#    - Giá trị hợp lệ: từ '01' đến '17' (theo dataset gốc Z24)
# --------------------------------------------
# Ví dụ full
# classes = ['01','03','04','05','06','07','09','10','11','12','13','14','15','16','17']
classes = ['01','03','04','05','06']  # chỉ chọn 5 lớp để chạy thử nghiệm

# --------------------------------------------
# 2. Chuẩn bị dữ liệu
#    - Trả về X_train, X_val, X_test, y_train, y_val, y_test, width
#    - width: độ dài mỗi chuỗi tín hiệu (ở đây = 65536 mẫu)
# --------------------------------------------
X_train, X_val, X_test, y_train, y_val, y_test, width = datasetManagement(classes, 65536)

# --------------------------------------------
# 3. Duyệt chọn model (.h5) qua hộp thoại thay vì hard-code đường dẫn
#    - Sử dụng tkinter.filedialog.askopenfilename để mở dialog chọn file
#    - Nếu không chọn, raise lỗi để tránh load nhầm
#    - Lưu ý: Cần môi trường có GUI (Windows/macOS/Linux với X11)
# --------------------------------------------
model_path = None
try:
    from tkinter import Tk, filedialog  # import tại chỗ để tránh lỗi ở môi trường headless
    Tk().withdraw()  # Ẩn cửa sổ gốc của Tkinter
    model_path = filedialog.askopenfilename(
        title="Chọn file mô hình WaveNet (.h5)",
        filetypes=[("H5 files", "*.h5"), ("All files", "*.*")]
    )
except Exception as e:
    # Trường hợp môi trường không hỗ trợ Tkinter (headless), có thể fallback sang nhập tay
    print(f"[WARN] Không mở được hộp thoại chọn file (.h5). Lý do: {e}")
    # Fallback: bạn có thể un-comment dòng dưới để nhập thủ công trong console:
    # model_path = input("Nhập đường dẫn đầy đủ tới file .h5: ").strip()

if not model_path:
    raise FileNotFoundError("Bạn chưa chọn (hoặc nhập) file .h5 nào! Dừng chương trình.")

print(f"Selected model file: {model_path}")

# --------------------------------------------
# 4. Load model
#    - compile=False: để tránh lỗi khác môi trường so với lúc train
#    - Sau đó compile lại với loss phù hợp
# --------------------------------------------
model = tf.keras.models.load_model(str(model_path), compile=False)
print(f"Loaded model from: {model_path}")

# --------------------------------------------
# 5. Xác định hàm loss phù hợp với định dạng nhãn
#    - Nếu y có one-hot (ndim>=2 và shape[-1] == num_classes) → categorical_crossentropy
#    - Nếu y chỉ là integer label (sparse) → sparse_categorical_crossentropy
# --------------------------------------------
num_classes = len(classes)

def pick_loss(y):
    if isinstance(y, np.ndarray) and y.ndim >= 2 and y.shape[-1] == num_classes:
        return "categorical_crossentropy"
    return "sparse_categorical_crossentropy"

selected_loss = pick_loss(y_train)

# --------------------------------------------
# 6. Compile lại model
#    - Optimizer: Adam với learning_rate=1e-4
#    - Loss: đã chọn ở trên
#    - Metric: accuracy
# --------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=selected_loss,
    metrics=["accuracy"]
)
print(f"Compiled with loss='{selected_loss}'.")

# --------------------------------------------
# 7. Đánh giá model trên tập test
#    - utils.evaluate_NN_models: trả về các chỉ số accuracy, precision, recall, f1
#    - confusion_matrix: để vẽ ma trận nhầm lẫn
# --------------------------------------------
result_all_models_FC, confusion_matrix_FC, predicitions_FC = utils.evaluate_NN_models(
    [model], X_test, y_test
)

# --------------------------------------------
# 8. In kết quả dưới dạng bảng
# --------------------------------------------
headers = ['models','accuracy','precision','recall','f1_score']
data_normal_FC = [["WaveNet"]]        # tên model
data_normal_FC.extend(result_all_models_FC)  # thêm kết quả đo được
utils.print_table(data_normal_FC, headers)

# --------------------------------------------
# 9. Vẽ confusion matrix để trực quan hoá độ chính xác phân lớp
# --------------------------------------------
utils.draw_confusion_matrix(confusion_matrix_FC, "WaveNet model")
