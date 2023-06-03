import cv2
import numpy as np

#잘린 문제들을 넣을 배열
question=[]



#화면 자르기 (중앙선 제일 위 s, 아래 e, 좌우 넓이 w)
def page_cut(page,s,e,w):
    left =  page[s+2:e, :w-2]
    right= page[s+2:e, w+2:w+w]
    
    return right, left

def createBground(image):
    
    height, width = image.shape[:2]
    backGround = np.zeros((height, width, 1), dtype=np.uint8)

    return backGround

#화면 자르기 전처리
def make(image):
    img = image
    h=[]
    w=[]
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(imgray,50,200,apertureSize = 3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,100,80)
    
    # cv2.imshow("gray",imgray)
    # cv2.imshow("edge", edges)
    # cv2.waitKey(0)
    
    # 라인 그어지는지 확인
    # bgimg = createBground(image)
    # for i in lines:
    #     cv2.line(bgimg, (int(i[0][0]), int(i[0][1])), (int(i[0][2]), int(i[0][3])), (255, 0, 0), 2)
    # dst = cv2.resize(bgimg, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("line",dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # 모의고사 상단 선, 중간 선 검출
    for i in range(len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            if y1 == y2:
                h.append([x1,y1])
                h.append([x2,y2])
            if x1 == x2:
                w.append([x1,y1])
                w.append([x2,y2])

    cv2.line(img,(h[0][0],h[0][1]),(h[1][0],h[1][1]),(0,0,255),1)
    cv2.line(img,(w[0][0],w[0][1]),(w[1][0],w[1][1]),(0,0,255),1)
    
    # 모의고사 상단선, 중간 선 검출 되었는지 확인
    # bgimg = createBground(image)
    # cv2.line(bgimg,(h[0][0],h[0][1]),(h[1][0],h[1][1]),255,1)
    # cv2.line(bgimg,(w[0][0],w[0][1]),(w[1][0],w[1][1]),255,1)
    # dst = cv2.resize(bgimg, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    
    # cv2.imshow('hough',dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # print((h[0][0],h[0][1]),(h[1][0],h[1][1]))
    # print((w[0][0],w[0][1]),(w[1][0],w[1][1]))
    
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
    return (contours)

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
    cv2.imshow("wrongimg",worng_img)
    cv2.waitKey(0)
    lines = cv2.HoughLinesP(dx,1, np.pi/180,50,minLineLength=80,maxLineGap = 200)
    print(lines)
    for i in lines:
        cv2.line(page_rl, (int(i[0][0]), int(i[0][1])), (int(i[0][2]), int(i[0][3])), (255, 0, 0), 2)
        # print(i[0])
        # cv2.imshow("a",page_rl)
        # cv2.waitKey(0)

    return lines
    # 추출된 라인의 rect을 만들기
    #29~35번째 줄 참고해서 만들고 좌표 저장하기()
   
def check_rec(r1, r2):
    tmp1 = r1
    tmp2 = r2
    
    if r1[1] < r1[3]:
        tmp1[3] = r1[1]
        tmp1[1] = r1[3]
        
    if r2[1] < r2[3]:
        tmp1[3] = r1[1]
        tmp1[1] = r1[3]
        
    max_lb_x = max(tmp1[0], tmp2[0])
    min_lb_y = min(tmp1[1], tmp2[1])
    min_rt_x = min(tmp1[2], tmp2[2])
    max_rt_y = max(tmp1[3], tmp2[3])
    if max_lb_x > min_rt_x or min_lb_y < max_rt_y:
        return False
    return True
    

def make_rec(lines):
    rect=[]
    for line in lines:
        flag = True
        if rect == None:
            rect = line[0]
        else:
            for r in rect:
                if check_rec(line[0],r):
                    flag = False
                if flag is False:
                    break
            if flag:
                rect.append(line[0])                
                
    return rect

#문제영역 표시
def find_qu_sangmin(count, rec, page):
    cnt_rst = []
    for c in count:
        flag = False
        x, y, w, h = cv2.boundingRect(c)
        rec_c = [x, y + h, x + w, y]
        for r in rec:
            print(r)
            print(rec_c)
            if check_rec(rec_c, r):
                flag = True
            if flag:
                cnt_rst.append(c)
                break
    cv2.drawContours(page, cnt_rst, -1, (0,0,255), 1)
    return cnt_rst #틀린문제 좌표

#원본, 채점 
origin=cv2.imread("./test/test18.png")
draw=src1 = cv2.imread("./test/KakaoTalk_20230603_210749181.jpg")
cv2.imshow('right',origin)
cv2.imshow('left', draw) 
cv2.waitKey(0)
 
#자르기   
right,left=make(origin)
d_right,d_left=make(draw)

cv2.imshow('right',right)
cv2.imshow('left',left)      

cv2.imshow('dright',d_right)
cv2.imshow('dleft',d_left)
cv2.waitKey(0)

count_l = find_question_area(left)
count_r = find_question_area(right)

line_r = find_wrong(d_right)
line_l = find_wrong(d_left)

rect_r = make_rec(line_r)
rect_l = make_rec(line_l)
print(rect_l)
print(rect_r)
find_qu_sangmin(count_l, rect_l, left)
find_qu_sangmin(count_r, rect_r, right)

cv2.imshow('right',right)
cv2.imshow('left',left)      

cv2.imshow('dright',d_right)
cv2.imshow('dleft',d_left)


cv2.waitKey(0)

