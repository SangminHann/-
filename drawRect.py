import cv2
import numpy as np
import detect as det

# import projectcode as pro


def drawRect(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 빨강색 범위
    lower_red = np.array([0, 10, 10])
    upper_red = np.array([10, 255, 255])

    # 이진화
    worng_img = cv2.inRange(img_hsv, lower_red, upper_red)

    # 블러처리
    blurred = cv2.GaussianBlur(worng_img, (3, 3), 0)

    # 소벨필터 적용 엣지검출(밝기변화)(엣지는 밝기의 변화가 있는 부분)
    # x 방향 미분 이미지(엣지의 수직 방향성 검출)
    gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    # y 방향 미분 이미지(엣지의 수평 방향성 검출)
    gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # 미분값 계산해서 변화량 크기 계산(엣지의 강도)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # 이미지 정규화(0~255)
    gradient_magnitude = cv2.normalize(
        gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # close 연산(잘린 객체들 이어붙이기)
    kernel = np.ones((3, 3), np.uint8)

    gradient_magnitude = cv2.morphologyEx(
        gradient_magnitude, cv2.MORPH_CLOSE, kernel, iterations=2
    )

    # contour찾기(o / x 를 찾아냄)

    contours, hierarchy = cv2.findContours(
        gradient_magnitude, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    selected_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 40:  # 적절한 threshold 값 설정(면적)
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


img = cv2.imread("problem.jpg")

result = drawRect(img)

cv2.imshow("result", result)
cv2.waitKey(0)
