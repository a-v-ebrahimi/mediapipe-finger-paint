import cv2
import mediapipe as mp
import time
import math
import numpy as np
from random import randrange

cac_logo = cv2.imread('banner.png', cv2.IMREAD_UNCHANGED)
thumb_pointing = cv2.imread('thumb_pointing.png', cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
lastPenTime = 0
penDown = 0
penDownZone = 0
penDownZoneHand = -1
sketch = []
sketches = []
sketchColor = (0, 0, 200)
clearCounter = 0

fingerTips = [4, 8, 12, 16, 20]
lastSketchX, lastSketchY = 0, 0
lastFingerPos = {}


def overlay_transparent(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_image

    return background


def angle_of_line(x1, y1, x2, y2):
    return math.degrees(math.atan2(-y2 + y1, x2 - x1))


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def paint_sketch(sketch_number, sketch_array):
    if len(sketch_array) > 10:
        # first smooth the curve
        s = 5
        kernel = np.ones(s)
        xx = [i[0] for i in sketch_array]
        yy = [i[1] for i in sketch_array]
        x = np.convolve(xx, kernel, 'valid') / s
        y = np.convolve(yy, kernel, 'valid') / s
        res = np.hstack((x[:, None], y[:, None]))

        # then paint it
        for t, s in enumerate(res):
            if t > 0:
                p = (int(res[t - 1][0]), int(res[t - 1][1]))
                pnt = (int(res[t][0]), int(res[t][1]))
                # color = (250, 250, 250 - sketch_number * 20)  # res[t][2]
                color = (50, 0, 20)  # res[t][2]
                # cv2.circle(img, pnt, 4, (255, 255, 255), cv2.FILLED)
                cv2.line(img, p, pnt, color, thickness=2)
                cv2.line(img, (w - p[0], p[1]), (w - pnt[0], pnt[1]), color, thickness=2)
                cv2.line(img, (w - p[0], h - p[1]), (w - pnt[0], h - pnt[1]), color, thickness=2)
                cv2.line(img, (p[0], h - p[1]), (pnt[0], h - pnt[1]), color, thickness=2)


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    overlay_transparent(img, cac_logo, 2, 540)
    # overlay_transparent(img, thumb_pointing, 0, 0)
    h, w, c = img.shape
    cv2.rectangle(img, (0, 0), (w, h), (100, 30, 60, 100))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handId, handLms in enumerate(results.multi_hand_landmarks):
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lastFingerPos[id] = (cx, cy)
                if id in fingerTips:
                    finger = fingerTips.index(id)
                    lm_prev = handLms.landmark[id - 1]
                    px, py = int(lm_prev.x * w), int(lm_prev.y * h)
                    angleOfRotation = angle_of_line(cx, cy, px, py) + 90
                    # print(cx, cy, px, py)
                    # print(angleOfRotation)
                    # if id > 0:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
                    if cx > w - 90 and id == 4 and penDown == 1 and handId == penDownZoneHand:
                        penDown = 0
                        if len(sketch) > 10:
                            sketches.append(sketch)
                            sketch = []
                    if id == 4 and cx < 90 and penDown == 0:
                        penDownZone = 1
                        penDownZoneHand = handId
                        penDown = 1
                    if id == 8:
                        lastSketchX = cx
                        lastSketchY = cy
                        if penDown == 1 and len(results.multi_hand_landmarks) == 1:
                            sketch.append((cx, cy, sketchColor))
                    if id == 4 and cx < lastSketchX - 20 and penDown == 1 and len(results.multi_hand_landmarks) == 1:
                        sketchColor = (randrange(255), randrange(255), randrange(255))
                    if id == 12 and cy < lastSketchY + 20 and len(results.multi_hand_landmarks) == 1:
                        clearCounter = clearCounter + 1
                        if clearCounter == 16:
                            clearCounter = 0
                            sketch = []
                            sketches = []
                            penDown = 0
                    if id == 20 and abs(cx - lastSketchX) > 50 and cy < lastSketchY + 20 and 4 in lastFingerPos and \
                            lastFingerPos[12][1] - lastFingerPos[id][1] > 30:
                        if time.time() - lastPenTime > 2:
                            lastPenTime = time.time()
                            if penDown == 1:
                                penDown = 0
                                clearCounter = 0
                                if len(sketch) > 10:
                                    sketches.append(sketch)
                                    sketch = []
                            else:
                                penDown = 1

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    cv2.putText(img, f"Pen Down : {penDown}", (30, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 255, 255), 2)
    cv2.putText(img, f"Segments : {len(sketches)}", (30, 80), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 255, 255), 2)
    if clearCounter > 0:
        cv2.putText(img, f"Clear? : {clearCounter}", (30, 110), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (0, 0, 255), 2)
    for skn, sk in enumerate(sketches):
        paint_sketch(skn, sk)
    paint_sketch(0, sketch)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS : {int(fps)}", (1150, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (110, 110, 110), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
