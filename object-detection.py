import numpy as np
import argparse
import cv2
import subprocess
import time
import os
from utils import img_process

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights')
    parser.add_argument('-cfg', '--config')
    parser.add_argument('-in', '--input')
    parser.add_argument('-o', '--output')
    parser.add_argument('-l', '--labels',)

    args = vars(parser.parse_args())

    labelsPath = args["labels"]
    labels = open(labelsPath).read().strip().split('\n')

    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    net = cv2.dnn.readNetFromDarknet(args["config"],args["weights"])

    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    if args["input"]:
        try:
            cap = cv2.VideoCapture(args["input"])
            height, width = None, None
            output = None
        except:
            raise ('[ERROR] Video input was not loaded please use relative path')
        finally:
           print ('[INFO] Processing Object detection this may take a few minutes')
        while True:
            rec, frame = cap.read()
            if width is None or height is None:
                height, width = frame.shape[:2]
            img_process(net, layer_names, height, width, frame, colors, labels)
            if output is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                output = cv2.VideoWriter(args["output"], fourcc, 30,
                    (frame.shape[1], frame.shape[0]), True)

            output.write(frame)

        print("[INFO] Detection Completed")
        output.release()
        cap.release()
