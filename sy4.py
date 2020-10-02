'''
统计灰度直方图(numpy)
# -*- coding: utf-8 -*
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

flag = False

def getAddr(param):
    return os.getcwd() + '\\sourses\\' + param

if __name__ == '__main__':
    
    img = cv2.imread(getAddr('flower2.jpg'), 0)
    
    cv2.imshow('a', img)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()
'''

#///////////////////////////////////////


'''
统计颜色直方图(cv2+numpy)
# -*- coding: utf-8 -*
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

flag = False

def getAddr(param):
    return os.getcwd() + '\\sourses\\' + param

if __name__ == '__main__':
    
    img = cv2.imread(getAddr('flower2.jpg'))
    
    color = ('b', 'g', 'r')
    
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0,256])
        plt.plot(histr, color = col)
        plt.xlim([0, 256])
    plt.show()
'''
    
#///////////////////////////////////////////

'''
累计直方图
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def getAddr(param):
    return os.getcwd() + '\\sourses\\' + param

if __name__ == '__main__':
    

    img = cv2.imread(getAddr('flowerx.png'), 0)

    #将数组变为一维
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    #计算累计分布图
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()
'''

#////////////////////////////////////////////////

'''
直方图均衡化
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def getAddr(param):
    return os.getcwd() + '\\sourses\\' + param    

cnt = 0
xi = [0, 0, 0, 0]
yi = [0, 0, 0, 0]

if __name__ == '__main__':
    
    img = cv2.imread(getAddr('fig5.jpg'), 0)
    equ = cv2.equalizeHist(img)
    res = np.hstack((img, equ))

    cv2.imshow('a', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''



#/////////////////////////
'''
思考题1
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def getAddr(param):
    return os.getcwd() + '\\sourses\\' + param


if __name__ == '__main__':
    
    img = cv2.imread(getAddr('Fig6.png'))
    tmp = cv2.imread(getAddr('Fig6.png'))
    cv2.imshow('origin', img)
    
    lower_blue=np.array([110,100,100])#blue
    upper_blue=np.array([130,255,255])
    lower_green=np.array([50,100,100])#green
    upper_green=np.array([70,255,255])
    lower_red=np.array([0,100,100])#red
    upper_red=np.array([10,255,255])

    mask_blue = cv2.inRange(tmp, lower_blue, upper_blue)
    mask_green = cv2.inRange(tmp, lower_green, upper_green)
    mask_red = cv2.inRange(tmp, lower_red, upper_red)
    green = cv2.bitwise_and(img, img, mask = mask_green)
    blue = cv2.bitwise_and(img, img, mask = mask_blue)
    red = cv2.bitwise_and(img, img, mask = mask_red)
    fin = green + blue + red
    

    res = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    td = cv2.split(res)
    #要处理的是td[2]即亮度分量
    #cv2.imshow('h', td[0])
    #cv2.imshow('s', td[1])
    #cv2.imshow('v', td[2])
    td[2] = cv2.equalizeHist(td[2])
    #均衡化后v通道
    #cv2.imshow('equ_v', td[2])
    equ_hsv = cv2.merge(td)
    equ_hsv = cv2.cvtColor(equ_hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('equ_hsv', equ_hsv)

    td = cv2.split(img)
    td[0] = cv2.equalizeHist(td[0])
    td[1] = cv2.equalizeHist(td[1])
    td[2] = cv2.equalizeHist(td[2])
    equ_rgb = cv2.merge(td)
    cv2.imshow('equ_rgb', equ_rgb)


    cv2.waitKey(0)
    cv2.destoryAllWindows()
'''

#//////////////////////////////////
'''
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

__wb__ = open('wb.txt', 'w')

def getAddr(param):
    return os.getcwd() + '\\sourses\\' + param


def prTxt(col, histr):
    print('follow : ' + col, file = __wb__)
    for j, k in enumerate(histr):
        print(col, end = '-', file = __wb__)
        print(j, end = '-', file = __wb__)
        print(k, file = __wb__)
    print('\n', file = __wb__)


def solve(param_1, param_2):
    lis = list(param_1)
    res = list(param_2)
    for i in range(1, 256):
        lis[i] = lis[i] + lis[i-1]
        res[i] = res[i] + res[i-1]
        
    
    i = 0
    j = 1
    cp = abs(res[0]-lis[0])
    while i < 256:
        while j < 256 and (abs(res[j-1]-lis[i]) > abs(res[j]-lis[i])):
            j += 1
        param_1[i] = j-1
        #print(j)
        i += 1

if __name__ == '__main__':
    
    fa = cv2.imread(getAddr('Fig6A.jpg'))
    sa = fa.shape
    fb = cv2.imread(getAddr('Fig6B.jpg'))
    sb = fa.shape

    fa = cv2.resize(fa, (int(sa[1]/2), int(sa[0]/2)))
    fb = cv2.resize(fb, (int(sb[1]/2), int(sb[0]/2)))

    #cv2.imshow('fa', fa)
    #cv2.imshow('fb', fb)

    color = ('b', 'g', 'r')

    histr_1 = [[0]*256 for i in range(3)]
    histr_2 = [[0]*256 for i in range(3)]
    
    for i, col in enumerate(color):
        histr = cv2.calcHist([fa], [i], None, [256], [0,256])
        histr_1[i] = histr
        #输出每个颜色个数到txt
        prTxt(col, histr)
        plt.plot(histr, color = col)
        plt.xlim([0, 256])
    plt.show()


    for i, col in enumerate(color):
        histr = cv2.calcHist([fb], [i], None, [256], [0,256])
        histr_2[i] = histr
        plt.plot(histr, color = col)
        plt.xlim([0, 256])
    plt.show()

    #处理直方图
    for i in range(3):
        solve(histr_1[i], histr_2[i])

    for i in range(3):
        plt.plot(histr_1[i], color = color[i])
        plt.xlim([0, 256])
    plt.show()

    td = cv2.split(fa)
    #np.set_printoptions(threshold=np.inf)
    #print(td[0], file = __wb__)
    
    mmin = 300
    mmax = -1
    for k in range(3):
        for i in range(len(td[k])):
            for j in range(len(td[k][i])):
                td[k][i][j] = histr_1[k][td[k][i][j]]
                
    res = cv2.merge(td)
    cv2.imshow('fa', fa)
    cv2.imshow('fb', fb)
    cv2.imshow('res', res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
'''