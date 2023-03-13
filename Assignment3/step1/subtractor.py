import cv2


def get_contour(img, min_c, flag=cv2.RETR_EXTERNAL):
    # Get contours
    contours = cv2.findContours(img, flag, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # noise points
    small_contours = []
    for con in contours:
        area = cv2.contourArea(con)
        if 0 < area < min_c:
            small_contours.append(con)

    # max contour: foreground
    max_contour = max(contours, key=cv2.contourArea)

    return img, contours, max_contour, small_contours


def get_frames(path, bg_path):
    capture = cv2.VideoCapture(path)
    bg_capture = cv2.VideoCapture(bg_path)

    if not capture.isOpened():
        print('Unable to open: ' + path)
        exit(0)

    backSub = cv2.createBackgroundSubtractorMOG2()

    i = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    while True:
        if i < 119:
            ret, frame = bg_capture.read()
        else:
            ret, frame = capture.read()

        if frame is None:
            break

        # 5 fps
        if i % 10 != 0 and i > 0:
            i = i + 1
            continue

        frame = cv2.resize(frame, [1000, 1000])
        mask = backSub.apply(frame, learningRate=0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        _, mask = cv2.threshold(mask, 200, 255, type=cv2.THRESH_BINARY)

        mask, cos, max_c, small_cs = get_contour((255 - mask), 2000, flag=cv2.RETR_TREE)
        mask = (255 - mask)
        cv2.drawContours(mask, small_cs, -1, (255, 255, 255), cv2.FILLED, maxLevel=6)

        if i > 119:
            cv2.imwrite("foreground/cam4/cam4_fg_" + str(i - 120) + ".png", mask)

        i = i + 1


get_frames("../4persons/video/cam4.avi", "../4persons/background/cam4.avi")
