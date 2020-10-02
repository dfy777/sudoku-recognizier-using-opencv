# -*- coding: utf-8 -*
'''
    图片旋转
import cv2
import numpy as np
import os

flag = False

def getAddr(param):
    return os.getcwd() + '\\sourses\\' + param

if __name__ == '__main__':
    
    img = cv2.imread(getAddr('flowers.tif'), 0)
    s = img.shape
        
    cv2.imshow('a', img)
        
    mat = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 45, 0.6)
    dst = cv2.warpAffine(img, mat, (s[1], s[0]))
        
    cv2.imshow('b', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''


'''
仿射变化
# -*- coding: utf-8 -*
import cv2
import numpy as np
import os

flag = False

def getAddr(param):
    return os.getcwd() + '\\sourses\\' + param

if __name__ == '__main__':
    
    img = cv2.imread(getAddr('drawing.png'))
    s = img.shape
    
    pt1 = np.float32([[50,50], [200,50], [50,200]])
    pt2 = np.float32([[10,100], [200,50], [100, 250]])
    
    mat = cv2.getAffineTransform(pt1, pt2)

    dst = cv2.warpAffine(img, mat, (s[1], s[0]))
    
    cv2.imshow('a', img)
    cv2.imshow('b', dst)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
        
        
'''
透视变化
import cv2
import numpy as np
import os

flag = False

def getAddr(param):
    return os.getcwd() + '\\sourses\\' + param

if __name__ == '__main__':
    
    img=cv2.imread(getAddr('paper.jpg'))
    s=img.shape
    pt1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pt2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

    mat= cv2.getPerspectiveTransform(pt1, pt2)
    
    dst = cv2.warpPerspective(img, mat, (int(s[1]/1), int(s[0]/2)))

    cv2.imshow('a', img)
    cv2.imshow('b', dst)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
    
    
'''
实验三 思考题
# -*- coding: utf-8 -*
import cv2
import numpy as np
import os

flag = False

def getAddr(param):
    return os.getcwd() + '\\sourses\\' + param

cnt = 0
xi = [0, 0, 0, 0]
yi = [0, 0, 0, 0]

if __name__ == '__main__':
    
    def click(event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            global cnt
            global xi, yi
            cnt = cnt+1
            if cnt <= 4:
                xi[cnt-1] = x
                yi[cnt-1] = y
    
    cv2.namedWindow('img1')
    cv2.setMouseCallback('img1', click)
    
    img = cv2.imread(getAddr('drawing.png'))
    
    while cnt <= 4:
        cv2.imshow('img1', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    cv2.namedWindow('img2')
    #img2 = np.zeros((512, 512, 3), np.uint8)
    
    s = img.shape
    
    pt1 = np.float32([[xi[0],yi[0]],[xi[1],yi[1]],[xi[2],yi[2]],[xi[3], yi[3]]])
    pt2 = np.float32([[0, 0],[0, s[1]],[s[0], 0],[s[1], s[0]]])
    
    mat= cv2.getPerspectiveTransform(pt1, pt2)
    dst = cv2.warpPerspective(img, mat, (512, 512))

    print(s[0], s[1])
    for i, val in enumerate(xi):
        print(i, end='')
        print(val, yi[i]) 

    
    cv2.imshow('img2', dst)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
'''
