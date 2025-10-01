import cv2
import numpy as np

def get_limits(color):

    c = np.uint8([[color]]) 
    hsvc = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = int(hsvc[0][0][0])

    lower = np.array([max(hue - 10, 0), 100, 100], dtype=np.uint8)
    upper = np.array([min(hue + 10, 179), 255, 255], dtype=np.uint8)
    return lower, upper

yellow = [0, 255, 255]
red    = [0, 0, 255]

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    mirrored = cv2.flip(frame, 1)
    hsvImage = cv2.cvtColor(mirrored, cv2.COLOR_BGR2HSV)

    # ---------- YELLOW ----------
    lower_y, upper_y = get_limits(yellow)
    mask_y = cv2.inRange(hsvImage, lower_y, upper_y)

    # ---------- RED (two hue bands) ----------
    lower_r1, upper_r1 = get_limits(red)
    lower_r2 = np.array([170, 100, 100], dtype=np.uint8)
    upper_r2 = np.array([180, 255, 255], dtype=np.uint8)

    mask_r1 = cv2.inRange(hsvImage, lower_r1, upper_r1)
    mask_r2 = cv2.inRange(hsvImage, lower_r2, upper_r2)
    mask_r  = cv2.bitwise_or(mask_r1, mask_r2)


    output = mirrored.copy()
    for mask, color_name, bgr in [(mask_y, 'Yellow', (0,255,255)),
                                  (mask_r, 'Red',    (0,0,255))]:
        mask_y = cv2.medianBlur(mask_y, 5)
        mask_r = cv2.medianBlur(mask_r, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, kernel)
        mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_CLOSE, kernel)
        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel)
        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(output, (x, y), (x+w, y+h), bgr, 2)
                cv2.putText(output, color_name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

    cv2.imshow('Detections', output)
    cv2.imshow('Red mask', mask_r)
    cv2.imshow('Yellow mask', mask_y)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


