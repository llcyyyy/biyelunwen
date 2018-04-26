import pickle
import numpy as np
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# csvlist = glob.glob('*.csv')
# for name in csvlist:
#     print(name)
#     with open(name, newline = '') as csvfile:
#         reader = csv.reader(csvfile)
#         dict = {}
#         for row in reader:
#             dict[row[0]] = eval(row[1])
#
#     dict['yolo_box'] = np.array(dict['yolo_box'])
#     dict['real_position'] = np.array(dict['real_position'])
#     name = name.split('.')[0]
#     with open(name +'.pkl', 'wb') as f:
#         pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)

X_all = np.empty(shape = (0, 3), dtype=np.float32)
y_all = np.empty(shape = (0, 2), dtype=np.float32)
os.chdir('E:\position transformation\position transformation')
track_list = glob.glob('*.pkl')
for name in track_list:
    with open(name, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        # bm = 0.5 * (data['yolo_box'][:, 0] + data['yolo_box'][:, 1])
        # xa = np.stack([bm, data['yolo_box'][:, 2]],axis=1)
        X_all = np.concatenate((X_all, data['yolo_box'][:, :3].astype(np.float32)))
        y_all = np.concatenate((y_all, data['real_position'][:, :2].astype(np.float32)))
X_train = X_all[:int(0.75 * X_all.shape[0]), :]
y_train = y_all[:int(0.75 * X_all.shape[0]), :]
X_valid = X_all[int(0.75 * X_all.shape[0]):, :]
y_valid = y_all[int(0.75 * X_all.shape[0]):, :]

plt.figure()
plt.plot(y_all[:, 0], y_all[:, 1], 'bo', markersize=1)
plt.show()