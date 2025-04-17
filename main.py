import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

print("Starting Air Calculator...")

digits_model = load_model("super_digit_classifier.keras")
operators_model = load_model("operators_model.keras")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

canvas = None
previous_position = None

DOT_IGNORE_AREA = 10
DOT_AREA_THRESHOLD = 40

def crop_and_predict_digit_or_dot(cropped):
    h, w = cropped.shape[:2]
    area = w*h
    if area < DOT_IGNORE_AREA:
        return None
    if area < DOT_AREA_THRESHOLD:
        return '.'
    max_side = max(w, h)
    scale = 20.0 / max_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out28 = np.zeros((28, 28), dtype=np.uint8)
    sx = (28 - new_w) // 2
    sy = (28 - new_h) // 2
    out28[sy:sy+new_h, sx:sx+new_w] = resized
    final_28 = out28.astype('float32') / 255.0
    x = final_28.reshape(1, 28, 28, 1)
    preds = digits_model.predict(x, verbose=1)  # verbose=1
    d = np.argmax(preds)
    return str(d)

def crop_and_predict_operator(cropped):
    h, w = cropped.shape[:2]
    max_side = max(w, h)
    scale = 20.0 / max_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out28 = np.zeros((28, 28), dtype=np.uint8)
    sx = (28 - new_w) // 2
    sy = (28 - new_h) // 2
    out28[sy:sy+new_h, sx:sx+new_w] = resized
    out28 = 255 - out28
    final_28 = out28.astype('float32') / 255.0
    x = final_28.reshape(1, 28, 28, 1)
    preds = operators_model.predict(x, verbose=1)
    idx = np.argmax(preds)
    op_map = {0: '+', 1: '/', 2: '*', 3: '-'}
    return op_map.get(idx, '?')

def preprocess_and_find_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_canvas = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    bin_dil = cv2.dilate(bin_canvas, kernel, iterations=1)
    contours, _ = cv2.findContours(bin_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return bin_dil, contours

def split_line_into_groups(boxes, space_x_thresh=15):
    boxes.sort(key=lambda b: b[0])
    sublines = []
    current = [boxes[0]]
    for i in range(1, len(boxes)):
        x,y,w,h = boxes[i]
        px,py,pw,ph = boxes[i-1]
        gap = x - (px+pw)
        if gap > space_x_thresh:
            sublines.append(current)
            current = [boxes[i]]
        else:
            current.append(boxes[i])
    if current:
        sublines.append(current)
    return sublines

def process_subgroup(bin_dil, boxes):
    boxes.sort(key=lambda b: b[0])
    s = ""
    for (x,y,w,h) in boxes:
        cropped = bin_dil[y:y+h, x:x+w]
        c = crop_and_predict_digit_or_dot(cropped)
        if c is not None:
            s += c
    if s.endswith('.'):
        s = s[:-1]
    if s == "":
        s = "0"
    return s

def process_operator_subgroup(bin_dil, boxes):
    boxes.sort(key=lambda b: b[0])
    x,y,w,h = boxes[0]
    cropped = bin_dil[y:y+h, x:x+w]
    return crop_and_predict_operator(cropped)

def predict_all_in_one_line(canvas_img):
    bin_dil, contours = preprocess_and_find_contours(canvas_img)
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        boxes.append((x,y,w,h))
    if not boxes:
        return
    sublines = split_line_into_groups(boxes, space_x_thresh=15)
    if len(sublines) == 3:
        # interpret as equation
        left_str = process_subgroup(bin_dil, sublines[0])
        op_str = process_operator_subgroup(bin_dil, sublines[1])
        right_str = process_subgroup(bin_dil, sublines[2])
        try:
            lv = float(left_str)
            rv = float(right_str)
            res = None
            if op_str == '+': res = lv + rv
            elif op_str == '-': res = lv - rv
            elif op_str == '*': res = lv * rv
            elif op_str == '/':
                if rv == 0: res = "ERR(div0)"
                else: res = lv / rv
            if isinstance(res, float) and not isinstance(res, str):
                if abs(res - round(res)) < 1e-9:
                    res = int(round(res))
            print(f"{left_str} {op_str} {right_str} = {res}")
        except:
            print(f"{left_str} {op_str} {right_str} = ERR")
    else:
        results = []
        for grp in sublines:
            val = process_subgroup(bin_dil, grp)
            results.append(val)
        print("; ".join(results))

def is_thumb_extended(landmarks):
    thumb_tip = landmarks.landmark[4]
    thumb_base = landmarks.landmark[2]
    return abs(thumb_tip.x - thumb_base.x) > 0.05

def is_hand_open(landmarks):
    tip_ids = [8,12,16,20]
    base_ids = [6,10,14,18]
    for tip_id, base_id in zip(tip_ids, base_ids):
        if landmarks.landmark[tip_id].y > landmarks.landmark[base_id].y:
            return False
    return True

def erase_area(landmarks, canvas_img, shape):
    try:
        palm_ids = [0,1,5,9,13,17]
        pls = [landmarks.landmark[i] for i in palm_ids]
        xs = [int(lm.x*shape[1]) for lm in pls]
        ys = [int(lm.y*shape[0]) for lm in pls]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cv2.rectangle(canvas_img, (x_min, y_min), (x_max, y_max), (0,0,0), -1)
    except:
        for fid in [8,12,16,20]:
            fx = int(landmarks.landmark[fid].x*shape[1])
            fy = int(landmarks.landmark[fid].y*shape[0])
            cv2.circle(canvas_img, (fx, fy), 20, (0,0,0), -1)

def main():
    global canvas, previous_position
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.8,
                        min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if canvas is None:
                canvas = np.zeros_like(frame)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                    fx = int(lm.landmark[8].x * frame.shape[1])
                    fy = int(lm.landmark[8].y * frame.shape[0])
                    if is_hand_open(lm):
                        erase_area(lm, canvas, frame.shape)
                        continue
                    if is_thumb_extended(lm):
                        if previous_position is not None:
                            cv2.line(canvas, previous_position, (fx, fy), (255,255,255), 5)
                        previous_position = (fx, fy)
                    else:
                        previous_position = None
            else:
                previous_position = None
            combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
            cv2.imshow("Air Draw Calculator", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                predict_all_in_one_line(canvas)
            elif key == ord('c'):
                canvas = np.zeros_like(frame)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
