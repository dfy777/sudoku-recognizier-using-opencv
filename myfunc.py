import cv2
import numpy as py
import os
import copy
import math

#将识别到的数独映射到大小为36*36的方块中
FIN_SIZE = 36
#数独的大小
SUDOKU_SIZE = 9
#切的数独成小方块后可能残留下线条
#需要去除边缘的几个像素
BOUND2CENTERZ_DIS = 5
#如果被切割的小块有效像素太少，可认为其中没有数组
MIN_ACTIVE_NUM = 70

def getAddr(param):
    return os.getcwd() + '\\' +param

def getOutAddr(param):
    return os.getcwd() + '\\output\\' + param

class RecImg:

    #初始化函数
    def __init__(self, name):
        self.sudoku = cv2.imread(getAddr(name))
        try:
            self.sudoku.shape
        except:
            print("fail to read " + name)
            return
        print("read success")

        self.update()

    #更新img函数
    def updateImg(self, newimg):
        self.sudoku = newimg
        try:
            self.sudoku.shape
        except:
            print("fail to update " + name)
            return
        print("update success")
        #自更新
        self.update()

    #自更新函数
    def update(self):
        shape = self.sudoku.shape
        self.high = shape[0]
        self.width = shape[1]
        self.iscolor = False

        if (len(shape) < 3):
            self.color_num = 0
            self.iscolor = False
        else:
            self.color_num = shape[2]
            self.iscolor = True

    #变为灰值图片
    def transGrid(self):
        self.sudoku = cv2.cvtColor(self.sudoku, cv2.COLOR_BGR2GRAY)
        self.update()

    #自适应二值化
    def binGrid(self, blocksize, constvalue):
        self.sudoku = cv2.adaptiveThreshold(self.sudoku, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blocksize, constvalue)
        self.update()

    #均值滤波
    def medianBlur(self, blocksize):
        self.sudoku = cv2.medianBlur(self.sudoku, blocksize)
        self.update()

    #高斯滤波
    def gaussBlur(self,blocksize):
        self.sudoku = cv2.GaussianBlur(self.sudoku,(blocksize,blocksize),0)
        self.update()

    #canny边缘提取
    def cannyBound(self,low,high,blocksize):
        self.sudoku = cv2.Canny(self.sudoku, threshold1 = low, threshold2 = high, apertureSize = 3)

    def show(self,str):
        cv2.imshow(str,self.sudoku)


#计算图像四个边角点坐标
def countBunglePoint(maxrec):
    bungle_point = [[0]*2 for i in range(4)]
    mmax = -1
    mmin = 1000**2
    mmaxi = -1
    mminj = -1
    for i in range(len(maxrec)):
        if (maxrec[i][0][0]+maxrec[i][0][1] > mmax):
            mmax = maxrec[i][0][0]+maxrec[i][0][1]
            mmaxi = i
        if (maxrec[i][0][0]+maxrec[i][0][1] < mmin):
            mmin = maxrec[i][0][0]+maxrec[i][0][1]
            mminj = i

    #print(bungle_point)
    #print(maxrec)
    #print(mmaxi)
    #print(mminj)
    #左上→右上→左下→右下
    bungle_point[0] = maxrec[mminj][0]
    bungle_point[3] = maxrec[mmaxi][0]

    for i in range(len(maxrec)):
        if i == mmaxi or i == mminj:
            continue
        #如果在右上角
        if abs(maxrec[i][0][0] - bungle_point[0][0]) < 50 and abs(maxrec[i][0][1] - bungle_point[3][1]) < 50:
                bungle_point[1] = maxrec[i][0]
        #如果在左下角
        if abs(maxrec[i][0][0] - bungle_point[3][0]) < 50 and abs(maxrec[i][0][1] - bungle_point[0][1]) < 50:
                bungle_point[2] = maxrec[i][0]
    
    return bungle_point

def testPoints(ori, points, name):
    img = copy.deepcopy(ori.sudoku)
    for i in range(4):
        cv2.circle(img, (points[i][0],points[i][1]),9,(0,0,255))
    cv2.imshow(name, img)


#x表示纵向高，y表示横向宽
#切割矩阵图中的9*9个小方块
#返回原图方块，二值化方块，方块有效像素数目
def divideBlock(x, y, img):
    block = img[x*FIN_SIZE:(x+1)*FIN_SIZE][:,y*FIN_SIZE:(y+1)*FIN_SIZE]
    block_thresh = cv2.cvtColor(block,cv2.COLOR_BGR2GRAY)
    block_thresh = cv2.adaptiveThreshold(block_thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 9)
    high, width = block_thresh.shape
    
    for i in range(high):
        for j in range(width):
            if (abs(i - high) <= BOUND2CENTERZ_DIS or
                    i <= BOUND2CENTERZ_DIS or
                    abs(j - width) <= BOUND2CENTERZ_DIS or
                    j <= BOUND2CENTERZ_DIS):
                block_thresh[i,j] = 0
    active_num = cv2.countNonZero(block_thresh)
    #findNumBoundRect(block_thresh,block,'numrect x-'+str(x)+' y-'+str(y)+'.jpg')
    #cv2.imshow('block x:'+str(x)+' y:'+str(y), block_thresh)
    #cv2.imwrite(getOutAddr('block x-'+str(x)+' y-'+str(y)+'.jpg'), block_thresh)
    return [block, block_thresh, active_num]

#利用boundingRect
#得到包含图片中包含数字的矩形
#返回矩阵左上角点左边，宽，高
def findNumBoundRect(img,ori):
    img, counter, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    max_bound_rect = []
    max_bound_rect_area = -1
    for i in range(len(counter)):
        bound_rect = cv2.boundingRect(counter[i])
        bound_rect_area = bound_rect[2]*bound_rect[3]
        if bound_rect_area > max_bound_rect_area:
            max_bound_rect_area = bound_rect_area
            max_bound_rect = bound_rect
    
    if len(max_bound_rect) == 0:
        return []

    x, y, w, h = max_bound_rect
    x = x-1
    y = y-1
    w = w+2
    h = h+2

    #cv2.rectangle(ori, (x,y), (x+w,y+h), (0,255,0), 2)
    #cv2.imshow(str,ori)
    #cv2.imwrite(getOutAddr(str_name), ori)
    return [x,y,w,h]


#识别方块中是否有数字
def recognizedNum(x, y, img, sudoku_img):
    [block, block_thresh, active_num] = divideBlock(x,y,img)
    if active_num < MIN_ACTIVE_NUM:
        return 0
    
    [xb,yb,w,h] = findNumBoundRect(block_thresh, block)
    number_rec = block_thresh[yb:yb+h,xb:xb+w]
    number_rec = cv2.resize(number_rec, (FIN_SIZE, FIN_SIZE), interpolation = cv2.INTER_LINEAR)
    sudoku_img.append(number_rec)

    return 1