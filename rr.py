import utils #library utils.py

from datasetManagement import datasetManagement #library DatasetManagement.py
from WaveNet import WavenetRun #library WaveNet.py
import tensorflow as tf
from pathlib import Path

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Get the list of available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Print information about each GPU
    for gpu in gpus:
        print("GPU:", gpu.name)
else:
    print("No GPU devices found.")

    