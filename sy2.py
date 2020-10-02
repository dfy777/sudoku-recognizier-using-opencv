# -*- coding: utf-8 -*
import cv2
import numpy as np


if __name__ == '__main__':
    drawing=False
    # 如果mode 为true 绘制矩形。按下'm' 变成绘制曲线。
    mode=1
    ix,iy = -1,-1
    # 创建回调函数
    def draw_circle(event,x,y,flags,param):
        global ix,iy,drawing,mode
        # 当按下左键是返回起始位置坐标
        if event==cv2.EVENT_LBUTTONDOWN:
            drawing=True
            ix,iy=x,y
        # 当鼠标左键按下并移动是绘制图形。event 可以查看移动，flag 查看是否按下
        elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
            if drawing==True:
                if mode == 1:
                    cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
                elif mode == 2:
                    r=int(np.sqrt((x-ix)**2+(y-iy)**2))
                    cv2.circle(img,(x,y),r,(0,0,255),-1)
                elif mode == 3:
                    cv2.line(img, (ix, iy), (x, y), (255,0,0))
                    
        # 当鼠标松开停止绘画。
        elif event==cv2.EVENT_LBUTTONUP:
            if mode == 1:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            elif mode == 2:
                cv2.circle(img,(x,y),5,(0,0,255),-1)
            elif mode == 3:
                cv2.line(img, (ix, iy), (x, y), (255,0,0))
    
    img=np.zeros((1024,1024,3),np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    while(1):
        cv2.imshow('image',img)
        k=cv2.waitKey(1)&0xFF
        if k == ord('a'):
            mode = 1
        elif k == ord('s'):
            mode = 2
        elif k == ord('d'):
            mode = 3
        elif k==27:
            break
    cv2.destroyAllWindows()

#=================================
# 图像乘法
import cv2
import numpy as np
import os

flag = False

def getAddr(param):
    return os.getcwd() + '\\sourses\\' + param

if __name__ == '__main__':
    
    img1 = cv2.imread(getAddr('woman.tif'), 0)
    
    res = cv2.multiply(img1, 1.5)

    cv2.imshow('a', img1)
    cv2.imshow('b', res)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

#=================================
#图片叠加
# -*- coding: utf-8 -*
import cv2
import numpy as np
import os

flag = False

def getAddr(param):
    return os.getcwd() + '\\sourses\\' + param

if __name__ == '__main__':
    
    img1 = cv2.imread(getAddr('hust.jpg'))
    img2 = cv2.imread(getAddr('hustmark.jpg'))
    
    tmp = img2.shape
    img2 = cv2.resize(img2, (int(tmp[1]/2), int(tmp[0]/2)))

    rows,cols,channels = img2.shape
    rows1, cols1, channels1 = img1.shape
    roi = img1[0:rows, cols1-cols:cols1]

    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask_front = cv2.threshold(img2gray, 50, 255, cv2.THRESH_BINARY) 
    mask_inv = cv2.bitwise_not(mask_front)

    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    img2_fg = cv2.bitwise_and(img2,img2,mask = mask_front)

    dst = cv2.add(img1_bg,img2_fg)
    img1[0:rows, cols1-cols:cols1 ] = dst  
    
    cv2.imshow('img1_bg',img1_bg)
    cv2.imshow('img2_fg',img2_fg)
    cv2.imshow('res',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cap.release()
    cv2.destroyAllWindows()
    
    

#=========================
#摄像头融合
# -*- coding: utf-8 -*
import cv2
import numpy as np
import os

flag = False

def getAddr(param):
    return os.getcwd() + '\\sourses\\' + param

if __name__ == '__main__':
    
    cap = cv2.VideoCapture(getAddr('vtest.avi'))
    cap_front = cv2.VideoCapture(getAddr('output1.avi'))
    
    def cfunc():
        global flag
        flag = not flag 
    
    def xiezi(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            cfunc()

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame',xiezi)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        ret_front, frame_front = cap_front.read()
        if ((not ret) or (not ret_front)):
            break
        if ret and ret_front:
            
            roi = frame.shape
            frame_front = cv2.resize(frame_front, (roi[1], roi[0]))
            
            
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame_front, cv2.COLOR_BGR2GRAY)
            ret, mask_front = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask_front)
            
            img1_bg = cv2.bitwise_and(frame,frame,mask = mask_front)
            img2_fg = cv2.bitwise_and(frame_front,frame_front,mask = mask_inv)
            
            dst = cv2.add(img1_bg, img2_fg)
            
            
            if flag == True: 
                font=cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(dst,'BANANICE!',(100,100), font, 4,(255,0,0),2)
            cv2.imshow('frame',dst)
            if cv2.waitKey(100) & 0xFF == ord('q')  :         
                break 
    cap.release()
    cv2.destroyAllWindows()

#==============================================
    #打开摄像头
    '''
    cap = cv2.VideoCapture(0)
    
    fourcc = cv2.cv.FOURCC('m', 'p', '4', 'v')
    out = cv2.VideoWriter('output1.avi', fourcc, 20.0, (640, 480))
    
    while(cap.isOpened):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame, 0)
            out.write(frame)
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    '''

