import numpy as np
import pandas as pd
import os, time, math, random
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import signal
from keras import backend as K 
from sklearn.utils import class_weight

from .data_processing import *
from .models import *
from .visualization import *

# Pathing and directories
datapath = '../dataset_fog_release/dataset/'
train_path = datapath + '../train_set.pkl'
test_path = datapath + '../test_set.pkl'
train_set = [""] * 10
test_set = [""] * 10
class_freqs = Counter({0:0, 1:0})

# Visualization
vis_path = os.path.join(datapath, 'S01R01.txt')
if not os.path.isfile(vis_path)
    x_plot(vis_path)

### Setup training and testing datasets ###
if os.path.isfile(train_path):
    with open(train_path, 'rb') as f:
        train_set = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_set = pickle.load(f)
else:
    for file_name in os.listdir(datapath):
        file = os.path.join(datapath, file_name)
        # Offset patient id to be 0-indexed
        pid = int(file_name[1:3].lstrip("0")) - 1

        raw_data = pd.read_csv(file, delim_whitespace=True, header = None)
        
        # Remove non experiment values
        raw_data = raw_data[raw_data[10] != 0]
        data = DataProcess(raw_data, pid)
        train_windows, train_targets = data.get_segments()
        test_windows, test_targets, freqs = data.get_testing_data()
        class_freqs += freqs

        if test_set[pid] != "":
            test_set[pid][0] = np.concatenate((test_set[pid][0], test_windows), axis=0)
            test_set[pid][1] = np.append(test_set[pid][1], test_targets)
        else:
            test_set[pid] = [test_windows, test_targets]

        if train_set[pid] != "":
            train_set[pid][0] = np.concatenate((train_set[pid][0], train_windows), axis=0)
            train_set[pid][1] = np.append(train_set[pid][1], train_targets)
        else:
            train_set[pid] = [train_windows, train_targets]

    class_weights = {0:class_freqs[1]/(class_freqs[1]+class_freqs[0]), 1:class_freqs[0]/(class_freqs[1]+class_freqs[0])}
    print("class weights", class_weights)

    with open(train_path, 'wb') as f:
        pickle.dump(train_set, f)
    with open(test_path, 'wb') as f:
        pickle.dump(test_set, f)

### Training ###
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

sensitivity_list = []
specificity_list = []
num_targets = 10

for target in range(0, num_targets):
    if target == 3 or target == 9:
        continue
    val_windows_ds = tf.data.Dataset.from_tensor_slices(test_set[target][0])
    val_targets_ds = tf.data.Dataset.from_tensor_slices(test_set[target][1])
    val_set_target = tf.data.Dataset.zip((val_windows_ds, val_targets_ds))
    val_set_target = val_set_target.shuffle(test_set[target][1].shape[0])
    val_set_target = val_set_target.batch(256)
    train_set_list = None

    # Setup so 0th dim is segment, 1st dim is target
    for i in range(len(train_set)):
        if i != target:
        if train_set_list is None:
            train_set_list = list(train_set[i])
        else:
            train_set_list[0] = np.concatenate((train_set_list[0], train_set[i][0]),axis=0)
            train_set_list[1] = np.append(train_set_list[1],train_set[i][1])

    train_windows_ds = tf.data.Dataset.from_tensor_slices(train_set_list[0])
    train_targets_ds = tf.data.Dataset.from_tensor_slices(train_set_list[1])
    train_set_target = tf.data.Dataset.zip((train_windows_ds, train_targets_ds))
    train_set_target = train_set_target.shuffle(train_set_list[1].shape[0])
    train_set_target = train_set_target.batch(256)

    K.clear_session()

    ### Model Setup ###
    model = LSTM_model()
    model.summary()

    if os.path.isfile("./models/model_{}".format(i)):
        model = keras.models.load_model("./models/model_{}".format(target))
    else:
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[sensitivity, specificity, "accuracy"])
        K.set_value(model.optimizer.learning_rate, 0.0001)

    history = model.fit(train_set_target, epochs=200, validation_data=val_set_target, class_weight=class_weights)
    sensitivity_list.append(history.history["val_sensitivity"][-1])
    specificity_list.append(history.history["val_specificity"][-1])
    model.save("./models/model_{}".format(target))

    K.clear_session()

# Final sensitivity/specificity plot
annotations=list(range(1, num_targets + 1))
plt.figure(figsize=(8,6))
plt.scatter(sensitivity_list, specificity_list, s=100, color="blue")
plt.xlabel("Sensitivity [%] Mean: {}%".format(round(sum(sensitivity_list)/len(sensitivity_list), 1)*100))
plt.ylabel("Specificity [%] Mean: {}%".format(round(sum(specificity_list)/len(specificity_list), 1)*100))

for i, label in enumerate(annotations):
    plt.annotate(label, (sensitivity_list[i], specificity_list[i]))
plt.savefig('sensitivity_specificity_graph.png', format='PNG', dpi=300)
plt.show()
