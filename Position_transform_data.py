import glob, os
import pickle
import numpy as np
dir_list = os.listdir('/home/chengyangli/U/pedestrian_prediction_dataset/train')
dir_list.sort()

for i in range(len(dir_list)):
    dir_list[i] = '/home/chengyangli/U/pedestrian_prediction_dataset/train' + '/' + dir_list[i] \
                  + '/left_rect/labels'

for track_index, path in enumerate(dir_list):
    os.chdir(path)
    # these boxes ar given as [x_middle, y_middle, width of box, height of box]
    body_box = []
    head_box = []
    real_position = []

    for data in glob.glob('*.txt'):
        with open(data) as f:
            for line in f:
                body_box.append(map(float, line.split()[4:8]))
                head_box.append(map(float, line.split()[13:17]))
                real_position.append(map(float, line.split()[-5:-2]))

    yolo_box = []
    # yolo box is given as [x_left, x_right, y_bottom, y_top] of a bounding box
    for b1, b2 in zip(body_box, head_box):
        # # combine head and body boxes
        # yolo_box.append([min(b1[0] - b1[2] / 2, b2[0] - b2[2] / 2),
        #             max(b1[0] + b1[2] / 2, b2[0] + b2[2] / 2),
        #             b1[1] - b1[3] / 2, b2[1] + b2[3] / 2])

        # only body boxes
        yolo_box.append([b1[0] - b1[2] / 2, b1[0] + b1[2] / 2,
                         b1[1] - b1[3] / 2, b1[1] + b1[3] / 2])

    pos_data = {'yolo_box': np.array(yolo_box),
                'real_position': np.array(real_position)}
    print track_index


    output_name = '/home/chengyangli/U/pedestrian_prediction_dataset/position transformation/track'\
                  + str(track_index) + '.pkl'
    with open(output_name, 'wb') as f:
        pickle.dump(pos_data, f, pickle.HIGHEST_PROTOCOL)


