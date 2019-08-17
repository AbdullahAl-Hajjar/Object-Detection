import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os

def draw(img, boxes, confidences, classids, idxs, colors, labels):
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            color = [int(c) for c in colors[classids[i]]]
            cv.rectangle(img, (x, y), (x+w, y+h), color, 1)
            text = "{}:  {:.0f} %".format(labels[classids[i]], round(confidences[i]*100, 2), 4)
            cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def  box(outputs, height, width, conf):
    boxes = []
    confidences = []
    classids = []
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            if confidence > conf:
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, box_width, box_height = box.astype('int')
                x = int(centerX - (box_width / 2))
                y = int(centerY - (box_height / 2))
                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                classids.append(classid)
    return boxes, confidences, classids

def img_process(net, layer_names, height, width, img, colors, labels):
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        outputs = net.forward(layer_names)
        end = time.time()
        boxes, confidences, classids = box(outputs, height, width, 0.5)
        idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        draw(img, boxes, confidences, classids, idxs, colors, labels)
