import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tabulate import tabulate

# ============================================================
# VẼ CONFUSION MATRIX
# ------------------------------------------------------------
# plot_confusion_matrix(ax, conf_matrix, title, cmap):
#   - ax:      đối tượng Axes (matplotlib) để vẽ lên
#   - conf_matrix: ma trận nhầm lẫn (ndarray 2D, kích thước [n_classes, n_classes])
#   - title:   tiêu đề của hình vẽ
#   - cmap:    tên colormap (chuỗi) cho seaborn.heatmap
# Chức năng: vẽ heatmap cho một confusion matrix duy nhất lên trục 'ax'
# ============================================================
def plot_confusion_matrix(ax, conf_matrix, title, cmap):
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap, cbar=False, ax=ax)
    ax.set_title(title)

# ------------------------------------------------------------
# draw_confusion_matrix(matrix, name):
#   - matrix: list các confusion matrix (mỗi phần tử là ndarray 2D)
#   - name:   tên mô hình/nhãn (không dùng trực tiếp trong code hiện tại)
# Lưu ý:
#   - Code hiện tại tạo 1 subplot (1 hàng, 1 cột), nhưng lại lặp qua nhiều
#     confusion matrix và vẽ chồng lên cùng một 'axes'.
#   - Điều này giữ nguyên hành vi gốc: kết quả cuối cùng hiển thị confusion
#     matrix cuối cùng trong danh sách 'matrix'.
#   - Nếu muốn vẽ nhiều confusion matrix cạnh nhau, cần tạo nhiều subplot.
# ------------------------------------------------------------
def draw_confusion_matrix(matrix, name):
    colormap = ["Blues", "Greens", "Oranges"]  # danh sách colormap mẫu
    fig, axes = plt.subplots(1, 1, figsize=(15, 5))  # chỉ một axes duy nhất
    for i in range(len(matrix)):
        # Vẽ lần lượt từng ma trận lên cùng một axes (chồng lên nhau)
        plot_confusion_matrix(axes, matrix[i], "Confusion Matrix  ", colormap[i])
    plt.tight_layout()

# ============================================================
# IN BẢNG KẾT QUẢ DẠNG TEXT
# ------------------------------------------------------------
# print_table(data, headers):
#   - data:    danh sách các CỘT (list of lists); hàm sẽ xoay (zip) thành HÀNG
#   - headers: danh sách tiêu đề cột
# Chức năng: in bảng kết quả đẹp bằng thư viện 'tabulate'
# ============================================================
def print_table(data, headers):
    table_data = list(zip(*data))  # chuyển danh sách cột thành danh sách hàng
    table = tabulate(table_data, headers=headers, tablefmt='grid')
    print(table)

# ============================================================
# ĐÁNH GIÁ MÔ HÌNH (EVALUATE_NN_MODELS)
# ------------------------------------------------------------
# evaluate_NN_models(model_list, X_test, y_test):
#   - model_list: danh sách các mô hình Keras đã compile/fit hoặc đã load
#   - X_test, y_test: dữ liệu và nhãn test
#
# Quy trình:
#   1) model.evaluate(...) -> lấy loss và accuracy cho từng model
#   2) model.predict(...)  -> dự đoán, rồi argmax theo trục lớp để lấy nhãn dự đoán
#   3) Tính precision, recall, f1 (average='weighted')
#   4) Tạo confusion_matrix cho từng model
#
# Trả về:
#   - [accuracy, precision, recall, f1_score]  : mỗi phần tử là list theo thứ tự model_list
#   - confusion_matrix_list                    : list các confusion matrix
#   - y_pred                                   : list các vector nhãn dự đoán cho từng model
#
# Lưu ý: Giữ nguyên biến đặt tên và logic như code gốc.
# ============================================================
def evaluate_NN_models(model_list, X_test, y_test):
    loss, accuracy, y_pred, precision, recall, f1_score, support, confusion_matrix_list = [], [], [], [], [], [], [], []
    for model in model_list:
        # 1) Đánh giá loss/accuracy
        loss_value, accuracy_value = model.evaluate(X_test, y_test)
        loss.append(loss_value)
        accuracy.append(accuracy_value)

        # 2) Dự đoán và lấy nhãn dự đoán (argmax theo trục lớp)
        y_pred_value = np.argmax(model.predict(X_test), axis=1)
        y_pred.append(y_pred_value)

        # 3) Lấy nhãn thật (ở đây giả định y_test là dạng integer labels)
        y_test_classes = y_test
        print("########", y_test_classes)  # debug: in nhãn thật
        print("$$$$$$$$", y_pred_value)    # debug: in nhãn dự đoán

        # 4) Tính precision, recall, f1 (trung bình 'weighted')
        precision_value, recall_value, f1_score_value, support_value = precision_recall_fscore_support(
            y_test_classes, y_pred_value, average='weighted'
        )
        precision.append(precision_value)
        recall.append(recall_value)
        f1_score.append(f1_score_value)
        support.append(support_value)

        # 5) Tạo confusion matrix và lưu lại
        confusion_matrix_value = confusion_matrix(y_test_classes, y_pred_value)
        confusion_matrix_list.append(confusion_matrix_value)

    return [accuracy, precision, recall, f1_score], confusion_matrix_list, y_pred

# ============================================================
# Utils: ĐỌC FILE HISTORY VÀ VẼ ACCURACY
# ------------------------------------------------------------
# Nhóm hàm hỗ trợ để:
#   - Tìm file history (*.txt) mới nhất trong thư mục ./History
#   - Đọc nội dung file history (dict dạng str) và parse lại thành dict Python
#   - Vẽ đường Accuracy (train/val) theo epoch
# ============================================================

# Tìm file history mới nhất trong thư mục
def find_latest_history_file(history_dir="./History"):
    import os, glob
    files = glob.glob(os.path.join(history_dir, "*.txt"))
    if not files:
        print(f"[WARN] Không tìm thấy file history trong: {history_dir}")
        return None
    # Sắp xếp theo thời gian sửa đổi, mới nhất trước
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

# Đọc dict từ file history (file được lưu bằng str(dict))
def load_history_dict(history_file):
    import ast
    with open(history_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
    # File được lưu dạng chuỗi biểu diễn dict -> parse an toàn bằng ast.literal_eval
    hist = ast.literal_eval(text)
    if not isinstance(hist, dict):
        raise ValueError("Nội dung history không phải dict.")
    return hist

# Vẽ accuracy (train/val) theo epochs; đồng thời cho phép lưu hình nếu cung cấp save_path
def plot_accuracy_from_history(history_dict, save_path=None, show_val=True):
    acc = history_dict.get("accuracy", [])       # danh sách accuracy train theo epoch
    val_acc = history_dict.get("val_accuracy", [])  # danh sách accuracy val theo epoch

    plt.figure()
    plt.plot(acc, label="Train Accuracy")
    if show_val and val_acc:
        plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.legend()
    if save_path:
        plt.savefig(save_path)  # lưu file PNG nếu cung cấp đường dẫn
    plt.show()  # hiển thị hình ra màn hình
