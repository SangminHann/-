import cv2
import numpy as np

#잘린 문제들을 넣을 배열
question=[]



#화면 자르기 (중앙선 제일 위 s, 아래 e, 좌우 넓이 w)
def page_cut(page,s,e,w):
    left =  page[s+2:e, :w-2]
    right= page[s+2:e, w+2:w+w]
    
    return right, left

#화면 자르기 전처리
def make(image):
    img=image
    h=[]
    w=[]
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(imgray,50,200,apertureSize = 3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,100)
    
    # 모의고사 상단 선, 중간 선 검출
    for i in range(len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            if y1 == y2:
                h.append([x1,y1])
                h.append([x2,y2])
            else:
                w.append([x1,y1])
                w.append([x2,y2])

    cv2.line(img,(h[0][0],h[0][1]),(h[1][0],h[1][1]),(0,0,255),1)
    cv2.line(img,(w[0][0],w[0][1]),(w[1][0],w[1][1]),(0,0,255),1)
    # cv2.imshow('hough',img)
    # print(h)
    # print(w)
    
    right, left =page_cut(img,max(h[1][1], h[2][1]),max(w[0][1], w[1][1]),w[0][0])
    
    return right, left

# 문제 영역 찾기(좌우잘린 원본문제)
def find_question_area(page_rl):
    
    imgray = cv2.cvtColor(page_rl, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(imgray, ksize=(3,3), sigmaX=0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    
    edge = cv2.Canny(imgray, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1000,100))
    closed = cv2.morphologyEx(edge, cv2.MORPH_DILATE, kernel)
    contours, hierarchy = cv2.findContours(closed.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #만들어진 좌우 페이지에 문제 영역 표시
    contours_image = cv2.drawContours(page_rl, contours, -1, (0,255,0), 1)
    
    for c in contours:

		#문제 영역 (x=왼쪽 젤위x 좌표, y=왼쪽 젤위 y좌표, h=높이,w=넓이)
        x,y,w,h = cv2.boundingRect(c)
        img_trim = page_rl[y:y+h , x:x+w]
        
        #question에 넣기 
        question.append(img_trim)

# /혹은 X모양 찾기(좌우로 잘린 풀이된 문제)  
def find_wrong(page_rl):
    
    img_hsv= cv2.cvtColor(page_rl, cv2.COLOR_BGR2HSV)
    
    # 빨강색 범위
    lower_red = np.array([-30, 100, 100])        
    upper_red = np.array([30, 255, 255])
    
    worng_img=cv2.inRange(img_hsv, lower_red, upper_red)
    
    #sobel
    dx = cv2.Sobel(worng_img, -1, 1, 0, scale=4)
    
    # #canny
    # eges = cv2.Canny(worng_img,0,0,apertureSize = 3)
    
    #일자 라인 추출
    lines = cv2.HoughLinesP(dx,1, np.pi/180,50,minLineLength=80,maxLineGap = 200)
    print(lines)
    for i in lines:
        cv2.line(page_rl, (int(i[0][0]), int(i[0][1])), (int(i[0][2]), int(i[0][3])), (255, 0, 0), 2)


    # 추출된 라인의 rect을 만들기
    #29~35번째 줄 참고해서 만들고 좌표 저장하기()
   
#원본, 채점 
origin=cv2.imread("./test/test1.png")
draw=src1 = cv2.imread("./test/draw.png")
 
#자르기   
right,left=make(origin)
d_right,d_left=make(draw)

#문제영역 표시
find_question_area(right)
find_question_area(left)

cv2.imshow('right',right)
cv2.imshow('left',left)

find_wrong(d_right)
find_wrong(d_left)

cv2.imshow('dright',d_right)
cv2.imshow('dleft',d_left)


cv2.waitKey(0)
