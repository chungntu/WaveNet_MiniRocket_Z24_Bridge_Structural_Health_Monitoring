import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from tabulate import tabulate

def plot_confusion_matrix(ax, conf_matrix, title, cmap):
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap, cbar=False, ax=ax)
    ax.set_title(title)

def draw_confusion_matrix(matrix,name):
  colormap=["Blues","Greens","Oranges"]
  fig, axes = plt.subplots(1, 1, figsize=(15, 5))
  for i in range(len(matrix)):
    plot_confusion_matrix(axes, matrix[i], "Confusion Matrix  ", colormap[i])
  plt.tight_layout()
  
#### function for plotting tables
def print_table(data,headers):
  table_data = list(zip(*data))
  table = tabulate(table_data, headers=headers, tablefmt='grid')
  print(table)

#function to calculate the size of machine learning models    
def evaluate_NN_models(model_list,X_test,y_test):
  loss, accuracy , y_pred ,precision, recall ,f1_score ,support ,confusion_matrix_list = [], [], [], [], [], [], [], []
  for model in model_list:
    loss_value,accuracy_value=model.evaluate(X_test, y_test)
    loss.append(loss_value)
    accuracy.append(accuracy_value)
    y_pred_value=np.argmax(model.predict(X_test), axis = 1)
    y_pred.append(y_pred_value)
    y_test_classes = y_test
    print("########",y_test_classes)
    print("$$$$$$$$",y_pred_value)
    precision_value, recall_value, f1_score_value, support_value = precision_recall_fscore_support(y_test_classes , y_pred_value, average= 'weighted' )
    precision.append(precision_value)
    recall.append(recall_value)
    f1_score.append(f1_score_value)
    support.append(support_value)
    confusion_matrix_value=confusion_matrix(y_test_classes, y_pred_value)
    confusion_matrix_list.append(confusion_matrix_value)
  return [accuracy ,precision, recall ,f1_score],confusion_matrix_list,y_pred



# ==========================
# Utils: đọc History và vẽ biểu đồ
# ==========================
def find_latest_history_file(history_dir="./History"):
    import os, glob
    files = glob.glob(os.path.join(history_dir, "*.txt"))
    if not files:
        print(f"[WARN] Không tìm thấy file history trong: {history_dir}")
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def load_history_dict(history_file):
    import ast
    with open(history_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
    # File được lưu bằng str(dict) -> parse bằng ast.literal_eval
    hist = ast.literal_eval(text)
    if not isinstance(hist, dict):
        raise ValueError("Nội dung history không phải dict.")
    return hist


# import matplotlib.pyplot as plt

def plot_accuracy_from_history(history_dict, save_path=None, show_val=True):
    acc = history_dict.get("accuracy", [])
    val_acc = history_dict.get("val_accuracy", [])

    plt.figure()
    plt.plot(acc, label="Train Accuracy")
    if show_val and val_acc:
        plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()
