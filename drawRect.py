import cv2
import numpy as np


def drawRect(image):
    ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    selected_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # 적절한 threshold 값 설정(면적)
            selected_contours.append(contour)

    bounding_rects = []
    for contour in selected_contours:
        # 시작좌표(x,y) 너비 w, 높이 h 검출(rect)
        rect = cv2.boundingRect(contour)
        bounding_rects.append(rect)

    square_rects = []
    for rect in bounding_rects:
        x, y, w, h = rect

        # contour를 둘러싼 rect의 한 변의 길이를 너비와 높이중 최대값으로 설정
        max_side = max(w, h)

        square_size = (max_side, max_side)

        # 중심좌표, 사각형 사이즈, 사각형 각도
        square_rect = ((x + w / 2, y + h / 2), square_size, 0)
        square_rects.append(square_rect)

    # 이미지에서 채첨부분 사각형만 잘라서 저장
    i = 1
    for rect in square_rects:
        points = cv2.boxPoints(rect)
        points = np.int0(points)
        # 선의 색상 : (0,255,0)(R,G,B) 선의두께 : 2
        cv2.drawContours(image, [points], 0, (0, 255, 0), 2)
        # 사각형의 시작 좌표와 너비, 높이를 구함
        x, y, w, h = cv2.boundingRect(points)
        # 원본 이미지에서 해당 부분만 잘라서 저장
        scoring_image = image[y : y + h, x : x + w]
        # 잘린 갯수만큼 파일로 저장
        cv2.imwrite(f"scoring_image{i}.jpg", scoring_image)
        i = i + 1
    return image
