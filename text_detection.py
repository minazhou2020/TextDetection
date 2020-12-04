import math
import os
import numpy as np
import cv2


def cropRotatedRect(img, rect):
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped


def resize_image(image):
    dimensions = image.shape
    height = int(dimensions[0] / 32) * 32
    width = int(dimensions[1] / 32) * 32
    cv2.resize(image, width, height)


def predict(output, detections, confidences, score_threshold):
    scores = output[0]
    geometry = output[1]
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]
            # If score is lower than threshold score, move to next x
            if (score < score_threshold):
                continue
            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]
            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]
            # Calculate offset
            offset = (
                [offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0], sinA * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    

def textDetection(image, rotated):
    # Read the deep learning network - tensorflow
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')
    # Saving a original image and shape
    orig = image.copy()
    # both image width and height should be multiples of 32
    resize_image(image)
    image_1 = image.copy()
    new_height, new_width = image.shape[:2]
    # construct a blob from the image to forward pass it to EAST model
    blob = cv2.dnn.blobFromImage(image, 1.0, (new_width, new_height), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")

    net.setInput(blob)
    output = net.forward(outputLayers)
    detections = []
    confidences = []
    score_threshold = 0.5
    predict(output, detections, confidences,score_threshold)
    indices = cv2.dnn.NMSBoxesRotated(detections, confidences, 0.1, 0.1)
    rois = []

    for i in indices:
        # get 4 corners of the rotated rect
        vertices = cv2.boxPoints(detections[i[0]])
        cropped = cropRotatedRect(image_1, detections[i[0]])
        rois.append(cropped)

    return rois


def main():
    path = "input\image\path"
    outPath = "output\image\path"
    # iterate through the names of contents of the folder
    for image_path in os.listdir(path):
        # skip file which extension is not '.jpg'
        if '.jpg' not in image_path:
            continue
        # create full input image path
        input_path = os.path.join(path, image_path)
        image = cv2.imread(input_path)
        # skip empty images
        if image is None:
            continue
        rois = textDetection(image)
        for i in range(0, len(rois)):
            full_path = os.path.join(outPath, 'detected_' + str(i) + '_' + image_path)
            if rois[i] is None:
                continue
            cv2.imwrite(full_path, rois[i])


if __name__ == '__main__':
    main()
