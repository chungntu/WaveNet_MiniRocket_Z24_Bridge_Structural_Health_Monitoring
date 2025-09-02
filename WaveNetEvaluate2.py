from datasetManagement import datasetManagement
import utils
import tensorflow as tf
from pathlib import Path
import numpy as np

#Insert here the classes you want to use for the training, the first element will be labeled as 0, the second as 1 and so on
#values from 01 to 17
#classes = ['01', '03', '04', '05', '06','07','09','10','11','12','13','14','15','16','17']
classes = ['01','03','04','05','06']

X_train, X_val, X_test, y_train, y_val, y_test, width = datasetManagement(classes, 65536)

models_dir = Path(r"C:\Users\Dell Precision 7810\Documents\GitHub\WaveNet_MiniRocket_Z24_Bridge_Structural_Health_Monitoring")
model_path = models_dir / "Wavenet8_1_9_65536_8.h5"

# load không compile, rồi compile lại theo kiểu nhãn
model = tf.keras.models.load_model(str(model_path), compile=False)
print(f"Loaded model from: {model_path}")

num_classes = len(classes)
def pick_loss(y):
    if isinstance(y, np.ndarray) and y.ndim >= 2 and y.shape[-1] == num_classes:
        return "categorical_crossentropy"
    return "sparse_categorical_crossentropy"

selected_loss = pick_loss(y_train)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=selected_loss,
              metrics=["accuracy"])
print(f"Compiled with loss='{selected_loss}'.")

# evaluate
result_all_models_FC, confusion_matrix_FC, predicitions_FC = utils.evaluate_NN_models([model], X_test, y_test)
headers = ['models','accuracy','precision','recall','f1_score']
data_normal_FC = [["WaveNet"]]
data_normal_FC.extend(result_all_models_FC)
utils.print_table(data_normal_FC, headers)
utils.draw_confusion_matrix(confusion_matrix_FC, "WaveNet model")
