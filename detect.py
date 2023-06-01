import cv2 as cv
import numpy as np

# 검정색 단일 채널 만들어주는 함수 검증용으로 만듬
def createBground(image):
    
    height, width = image.shape[:2]
    backGround = np.zeros((height, width, 1), dtype=np.uint8)

    return backGround


# 빨간색으로 Threshold 하는 작업
def thresRed(img_path):
    
    image = cv.imread(img_path)
    b, g, r = cv.split(image)
    
    # 순수 빨간색 영역을 뽑아내는 작업
    # R채널에서 흰색도 빨간색 취급하기 때문에
    # B G 채널에서의 값을 빼줌
    bg = cv.add(b, g)
    r = cv.subtract(r, bg)
    
    # 원하는 값 잘 안나오면 20에서 고치면 됨
    ret, rst = cv.threshold(r, 20, 255, cv.THRESH_BINARY)
    
    # threshold 잘 됐는지 확인
    # cv.imshow("thresRed", rst)
    # cv.waitKey(0)
    
    return rst


# 허프 변환
def detectLines(img_path):
    
    thres = thresRed(img_path)
    
    # 노이즈를 제거 선 검출의 일관성 높임
    # 3번째 인자에는 표준편차가 들어가는데 0을 넣으면 자동 계산
    blurred = cv.GaussianBlur(thres, (5, 5), 0)

    # 테두리 검출
    edges = cv.Canny(blurred, 50, 200)
    
    # 확률적 허프변환 함수
    # 0 edge 이미지
    # 1 누설선의 거리 분해능 보통 1로함
    # 2 각도 분해능 보통 1도로 함
    # 3 여기서부턴 노가다 직선으로 판단할 점 개수
    # 4 검출되는 선의 최소 길이
    # 5 하나의 직선으로 간주하기 위한 최대 허용 간격 4,5 는 생략가능
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 30, 20, 10)
    
    # 좌표가 잘 뽑혔는지 확인
    # back = createBground(edges)
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv.line(back, (x1, y1), (x2, y2), 255, 2)
    # cv.imshow("detectLines", back)
    # cv.waitKey(0)
    return lines