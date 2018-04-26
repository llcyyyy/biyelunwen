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

input = tf.placeholder(tf.float32, (None, 3), 'Yolo_box')
output = tf.placeholder(tf.float32, (None, 1), 'Real_position')
X_all = np.empty(shape = (0, 3), dtype=np.float32)
y_all = np.empty(shape = (0, 1), dtype=np.float32)
os.chdir('E:\position transformation\position transformation')
track_list = glob.glob('*.pkl')
for name in track_list:
    with open(name, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        # bm = 0.5 * (data['yolo_box'][:, 0] + data['yolo_box'][:, 1])
        # xa = np.stack([bm, data['yolo_box'][:, 2]],axis=1)
        X_all = np.concatenate((X_all, data['yolo_box'][:, :3].astype(np.float32)))
        y_all = np.concatenate((y_all, data['real_position'][:, 0:1].astype(np.float32)))
X_train = X_all[:int(0.75 * X_all.shape[0]), :]
y_train = y_all[:int(0.75 * X_all.shape[0]), :]
X_valid = X_all[int(0.75 * X_all.shape[0]):, :]
y_valid = y_all[int(0.75 * X_all.shape[0]):, :]

# Normalization
X_train = (X_train - np.mean(X_train, axis=0)) / [1000, 1000, 200]
y_train = (y_train - np.mean(y_train, axis=0)) / 300
X_valid = (X_valid - np.mean(X_valid, axis=0)) / [1000, 1000, 200]
y_valid = (y_valid - np.mean(y_valid, axis=0)) / 300

cell_num = [1000, 800, 500]

# network architectur
def my_network(x):
    fc1_W = tf.get_variable(shape=(3, cell_num[0]), initializer =tf.contrib.layers.xavier_initializer(), name='fc1w')
    fc1_b = tf.get_variable(name='fc1b', shape=(cell_num[0]), initializer=tf.zeros_initializer)
    fc1 = tf.matmul(x, fc1_W) + fc1_b

    fc1 = tf.nn.tanh(fc1)

    fc2_W = tf.get_variable(shape=(cell_num[0], cell_num[1]), initializer=tf.contrib.layers.xavier_initializer(), name='fc2w')
    fc2_b = tf.get_variable(name='fc2b', shape=(cell_num[1]), initializer=tf.zeros_initializer)
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    fc2 = tf.nn.tanh(fc2)

    fc3_W = tf.get_variable(shape=(cell_num[1], cell_num[2]), initializer=tf.contrib.layers.xavier_initializer(), name='fc3w')
    fc3_b = tf.get_variable(name='fc3b', shape=(cell_num[2]), initializer=tf.zeros_initializer)
    fc3 = tf.matmul(fc2, fc3_W) + fc3_b

    fc3 = tf.nn.tanh(fc3)

    fc4_W = tf.get_variable(shape=(cell_num[2], 1), initializer=tf.contrib.layers.xavier_initializer(), name='fc4w')
    fc4_b = tf.get_variable(name='fc4b', shape=(1), initializer=tf.zeros_initializer)
    fc4 = tf.matmul(fc3, fc4_W) + fc4_b

    return fc4

def evaluate(x, y):
    sess = tf.get_default_session()
    return sess.run(el_loss, feed_dict={input: x, output: y})

epochs = 100
learning_rate = 0.0001
loss_log = []

predicts = my_network(input)
loss_operation = tf.losses.mean_squared_error(output, predicts)
el_loss = tf.reduce_mean(tf.abs(predicts - output), axis = 0)
optimizer = tf.train.AdamOptimizer(learning_rate)
training_operation = optimizer.minimize(loss_operation)

batch_size = 210


from sklearn.utils import shuffle
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        # for offset in range(0, num_examples, batch_size):
        #     end = offset + batch_size
        #     batch_x, batch_y = X_train[offset:end], y_train[offset:end]

        sess.run(training_operation, feed_dict={input: X_train, output: y_train})

        # validation_accuracy = evaluate(X_valid, y_valid)
        # train_accuracy = evaluate(X_train, y_train)

        total_loss = sess.run(loss_operation, feed_dict={input: X_train, output: y_train})
        valid_loss = sess.run(loss_operation, feed_dict={input: X_valid, output: y_valid})
        loss_log.append(valid_loss)
        jb = sess.run(predicts, feed_dict={input: X_valid, output: y_valid})
        # valid_loss = sess.run(loss_operation, feed_dict={x: X_valid, y: y_valid})
        print("EPOCH {} ...".format(i + 1))
        # print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Total train loss:", total_loss)
        print("Total validation loss:", valid_loss*300)
        print("Validation loss for x, y:", 300*evaluate(X_valid, y_valid))
        # print("Train accuracy = {:.3f}:".format(train_accuracy))
        print()

plt.figure()
plt.plot(range(epochs), loss_log)
plt.figure()
plt.plot(jb[:, 0], 'bo', markersize = 1)
plt.plot(y_valid[:, 0], 'ro', markersize = 1)
plt.show()