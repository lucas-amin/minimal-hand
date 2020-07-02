from wrappers import ModelPipeline, ModelDet
from utils import *
import config
import matplotlib.pyplot as plt
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


model = ModelDet(config.DETECTION_MODEL_PATH)

cap = cv2.VideoCapture(0)

while 1:
    ret, frame_large = cap.read()

    if frame_large is None:
        continue

    if frame_large.shape[0] > frame_large.shape[1]:
        margin = int((frame_large.shape[0] - frame_large.shape[1]) / 2)
        frame_large = frame_large[margin:-margin]
    else:
        margin = int((frame_large.shape[1] - frame_large.shape[0]) / 2)
        frame_large = frame_large[:, margin:-margin]

    frame_large = np.flip(frame_large, axis=1).copy()
    frame = cv2.resize(frame_large, (128, 128))

    result_3d, heatmap = model.process(frame)

    fig = plt.figure(1)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, projection='3d')
    ax1.imshow(frame)
    # plot_hand(coord_hw, ax1)

    ax2.imshow(frame)
    # plot_hand(coord_hw_crop, ax2)
    ax3.imshow(frame)

    plot_hand_3d(heatmap, ax4)
    ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
    ax4.set_xlim([0, 32])
    ax4.set_ylim([0, 32])
    ax4.set_zlim([0, 32])
    plt.show()
