import cv2
import numpy as np
import detect as det


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


# result1 = det.thresRed("problem.jpg")
# cv2.imshow("result1", result1)

# image = cv2.imread("problem.jpg")
# image = cv2.resize(image, (0, 0), fx=0.7, fy=0.7)
# cv2.imshow("origin", image)

# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# lower_red = np.array([0, 10, 10])
# upper_red = np.array([10, 255, 255])

# thresholded = cv2.inRange(hsv_image, lower_red, upper_red)

# blurred = cv2.GaussianBlur(thresholded, (3, 3), 0)

# gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
# gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

# gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

# gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# kernel = np.ones((5,5), np.uint8)
# # cv2.imshow("result,",gradient_magnitude)

# lines = cv2.HoughLinesP(gradient_magnitude, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)

# # x와 /만 검출
# x_lines = []
# slash_lines = []

# if lines is not None:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

#         # 특정 각도 범위에 있는 선을 x 또는 /로 분류
#         if (angle > 45 and angle < 135) or (angle < -45 and angle > -135):
#             x_lines.append(line)
#         elif (angle > -45 and angle < 45) or (angle < -135 and angle > -180) or (angle > 135 and angle < 180):
#             slash_lines.append(line)

# # 검출된 선 그리기
# for line in x_lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(gradient_magnitude, (x1, y1), (x2, y2), (0, 255, 0), 2)

# for line in slash_lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(gradient_magnitude, (x1, y1), (x2, y2), (0, 0, 255), 2)

# cv2.imshow("result,",gradient_magnitude)


# dilated_image = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=2)

# ret, binary = cv2.threshold(dilated_image, 127, 255, cv2.THRESH_BINARY)

# contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# selected_contours = []
# for contour in contours:
#     area = cv2.contourArea(contour)
#     if area > 50:  # 적절한 threshold 값 설정
#         selected_contours.append(contour)

# bounding_rects = []
# for contour in selected_contours:
#     rect = cv2.boundingRect(contour)#xywh
#     bounding_rects.append(rect)

# square_rects = []
# for rect in bounding_rects:
#     x,y,w,h = rect

#     max_side = max(w,h)

#     square_size = (max_side, max_side)

#     square_rect = ((x+w/2,y+h/2), square_size, 0)
#     square_rects.append(square_rect)

# i = 1
# for rect in square_rects:
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)
#     cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
#     x, y, w, h = cv2.boundingRect(box)
#     cropped_image = image[y:y+h, x:x+w]
#     cv2.imwrite(f'cropped_image{i}.jpg', cropped_image)
#     i=i+1

# cv2.imshow("dilate_image", dilated_image)
# cv2.imshow("new_image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
