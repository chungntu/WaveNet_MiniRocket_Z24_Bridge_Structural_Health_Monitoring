import os
import numpy as np
import scipy.io
import tensorflow as tf
import random

# ------------------------------------------------------------
# Đặt seed cho NumPy và random để đảm bảo tái lập (reproducibility)
# Nhờ vậy việc chia tập dữ liệu train/val/test sẽ luôn giống nhau
# ------------------------------------------------------------
np.random.seed(4)
random.seed(4)

from sklearn.model_selection import train_test_split

# ============================================================
# datasetManagement(classes, windows_length=65536)
# ------------------------------------------------------------
# Hàm xử lý dữ liệu thô thành dữ liệu huấn luyện cho WaveNet
# Input:
#   - classes: danh sách các lớp (tên thư mục con, ví dụ ['01','03',...])
#   - windows_length: độ dài chuỗi đầu vào (mặc định = 65536)
# Output:
#   - X_train, X_val, X_test: dữ liệu đầu vào (TensorFlow tensor, shape [N, windows_length, 1])
#   - y_train, y_val, y_test: nhãn tương ứng
#   - width: chiều dài 1 chuỗi (windows_length)
# ============================================================
def datasetManagement(classes, windows_length=65536):
    if classes is None:
        raise ValueError("Classes cannot be None.")        

    # --------------------------------------------------------
    # Tạo danh sách đường dẫn tới từng thư mục class
    # --------------------------------------------------------
    classesPaths = []
    for classNum in classes:
        classesPaths.append('./DatasetPDT/'+classNum+'/avt/')
    print(classesPaths)

    # Độ dài chuỗi time series mặc định
    lenTimeserie = 65536

    # --------------------------------------------------------
    # Đọc dữ liệu từ các file .mat trong từng thư mục class
    # --------------------------------------------------------
    datasetFull = []   # lưu toàn bộ time series
    labelsFull = []    # lưu nhãn class tương ứng
    for classPath in classesPaths:
        print(classPath)
        for path in os.listdir(classPath):
            if path.endswith(".mat"):
                # Load file .mat
                mat = scipy.io.loadmat(classPath+path)
                dataAggregated = mat['data']
                dataAggregated = dataAggregated.T  # transpose: mỗi hàng là một mẫu

                for i in range(len(dataAggregated)):
                    # Nếu chuỗi ngắn hơn 65536 thì pad bằng giá trị cuối
                    if(len(dataAggregated[i]) < lenTimeserie):
                        npData = np.array(dataAggregated[i])
                        last_value = npData[-1]
                        additional_values = np.full(lenTimeserie - len(npData), last_value)
                        npData = np.concatenate((npData, additional_values))                        
                        datasetFull.append(np.array(npData))
                    else:
                        # Nếu đủ dài thì thêm trực tiếp
                        datasetFull.append(np.array(dataAggregated[i]))
                    # Gán nhãn = chỉ số class
                    labelsFull.append(classesPaths.index(classPath))

    # --------------------------------------------------------
    # Trộn dữ liệu (shuffle)
    # --------------------------------------------------------
    combined_data = list(zip(datasetFull, labelsFull))
    from sklearn.utils import shuffle
    shuffled_data = shuffle(combined_data, random_state=42)
    combined_data = None
    # Tách lại thành dữ liệu và nhãn
    shuffled_dataset, shuffled_labels = zip(*shuffled_data)
    shuffled_dataset = np.array(shuffled_dataset)
    shuffled_labels = np.array(shuffled_labels)

    # --------------------------------------------------------
    # Chuẩn hóa dữ liệu bằng StandardScaler (zero mean, unit variance)
    # --------------------------------------------------------
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_dataset = scaler.fit_transform(
        shuffled_dataset.reshape(-1, lenTimeserie)
    ).reshape(shuffled_dataset.shape)

    # --------------------------------------------------------
    # Chia theo từng class: 70% train, 20% test, 10% val
    # --------------------------------------------------------
    combined_data = list(zip(scaled_dataset, shuffled_labels))
    train_data, test_data, val_data = [], [], []

    for class_label in range(len(classes)):
        # Lấy mẫu cùng class
        class_data, class_labels = zip(*[(data, label) for data, label in combined_data if label == class_label])
        # Chia 80% train, 20% test
        train, test, ytrain, ytest = train_test_split(class_data, class_labels, test_size=1/5, random_state=42)
        # Chia tiếp trong train: 12.5% cho val (tức 1/8)
        train, val, ytrain, yval = train_test_split(train, ytrain, test_size=1/8, random_state=42)
        train_data.extend(zip(train, ytrain))
        test_data.extend(zip(test, ytest))
        val_data.extend(zip(val, yval))

    # --------------------------------------------------------
    # Chuyển dữ liệu sang numpy array
    # --------------------------------------------------------
    X_train_pre, y_train_pre = map(np.array, zip(*train_data))
    X_test_pre, y_test_pre = map(np.array, zip(*test_data))
    X_val_pre, y_val_pre = map(np.array, zip(*val_data))

    # --------------------------------------------------------
    # Tạo windows (ở đây windows_length = 65536, tức là lấy trọn chuỗi)
    # --------------------------------------------------------
    def create_windows(data_array, label_array, windows_length):
        X, y = [], []
        for i in range(len(data_array)):
            for start in range(0, len(data_array[i]) - windows_length + 1, windows_length):
                end = start + windows_length
                X.append(data_array[i][start:end])
                y.append(label_array[i])
        return X, y

    X_train, y_train = create_windows(X_train_pre, y_train_pre, windows_length)
    X_val, y_val = create_windows(X_val_pre, y_val_pre, windows_length)
    X_test, y_test = create_windows(X_test_pre, y_test_pre, windows_length)
    
    width = len(X_train[0])  # độ dài chuỗi sau khi cắt window (ở đây = 65536)

    # --------------------------------------------------------
    # Chuyển sang numpy array và reshape thành [N, width, 1]
    # --------------------------------------------------------
    X_train = np.array([np.array(x) for x in X_train])
    X_val = np.array([np.array(x) for x in X_val])
    X_test = np.array([np.array(x) for x in X_test])

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1) 
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)     
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    print(X_train.shape, X_val.shape, X_test.shape)   

    # --------------------------------------------------------
    # Đổi sang TensorFlow tensor với dtype float32
    # --------------------------------------------------------
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    return X_train, X_val, X_test, y_train, y_val, y_test, width
