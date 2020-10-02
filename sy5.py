#//////////////////////////////////
'''
自定义矩阵 滤波
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

__wb__ = open('wb.txt', 'w')

def getAddr(param):
    return os.getcwd() + '\\sourses\\' + param

if __name__ == '__main__':
    
    img = cv2.imread(getAddr('lena.bmp'))
    kernel = np.ones((5,5),np.float32)/25
    #cv.filter2D(src, dst, kernel, anchor=(-1, -1))
    #ddepth –desired depth of the destination image;
    #if it is negative, it will be the same as src.depth();
    #the following combinations of src.depth() and ddepth are supported:
    #src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
    #src.depth() = CV_16U/CV_16S, ddepth = -1/CV_32F/CV_64F
    #src.depth() = CV_32F, ddepth = -1/CV_32F/CV_64F
    #src.depth() = CV_64F, ddepth = -1/CV_64F
    #when ddepth=-1, the output image will have the same depth as the source.
    dst = cv2.filter2D(img,-1,kernel)
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
    plt.xticks([]), plt.yticks([])
    plt.show()
'''

#/////////////////////////////
'''
blur滤波
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

__wb__ = open('wb.txt', 'w')

def getAddr(param):
    return os.getcwd() + '\\sourses\\' + param

if __name__ == '__main__':
    
    img = cv2.imread(getAddr('lena.bmp'))
    blur3 = cv2.blur(img,(3,3))
    blur5 = cv2.blur(img,(5,5))
    blur7 = cv2.blur(img,(7,7))
    plt.subplot(221),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(blur3),plt.title('Blurred3*3')
    plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(blur5),plt.title('Blurred 5*5')
    plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(blur7),plt.title('Blurred 7*7')
    plt.xticks([]), plt.yticks([])
    plt.show()
'''

#//////////////////////////////
'''
中值滤波 
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

flag = False

def getAddr(param):
    return os.getcwd() + '\\sourses\\' + param

if __name__ == '__main__':
    img = cv2.imread(getAddr('lena.bmp'))
    blur3 = cv2.medianBlur(img, 3)
    blur5 = cv2.medianBlur(img, 5)
    blur7 = cv2.medianBlur(img, 7)
    plt.subplot(221),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(blur3),plt.title('Blurred 3*3')
    plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(blur5),plt.title('Blurred 5*5')
    plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(blur7),plt.title('Blurred 7*7')
    plt.xticks([]), plt.yticks([])
    plt.show()
'''

#//////////////////
'''
高斯滤波
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

flag = False

def getAddr(param):
    return os.getcwd() + '\\sourses\\' + param

if __name__ == '__main__':
    img = cv2.imread(getAddr('lena.bmp'))
    blur3 = cv2.GaussianBlur(img, (3,3), 0)
    blur5 = cv2.GaussianBlur(img, (5,5), 0)
    blur7 = cv2.GaussianBlur(img, (7,7), 0)
    plt.subplot(141),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(142),plt.imshow(blur3),plt.title('Blurred 3*3')
    plt.xticks([]), plt.yticks([])
    plt.subplot(143),plt.imshow(blur5),plt.title('Blurred 5*5')
    plt.xticks([]), plt.yticks([])
    plt.subplot(144),plt.imshow(blur7),plt.title('Blurred 7*7')
    plt.xticks([]), plt.yticks([])
    plt.show()

    
'''

#//////////////////////////////////
'''
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

flag = False

def getAddr(param):
    return os.getcwd() + '\\sourses\\' + param

if __name__ == '__main__':
    img = cv2.imread(getAddr('moon.jpg'))
    res = cv2.imread(getAddr('moon.jpg'))
    
    #cv2.CV_64F 输出图像的深度（数据类型），可以使用-1, 与原图像保持一致 np.uint8
    laplacian=cv2.Laplacian(img,cv2.CV_64F)
    # 参数 1,0 为只在 x 方向求一阶导数，最大可以求 2 阶导数。
    sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    # 参数 0,1 为只在 y 方向求一阶导数，最大可以求 2 阶导数。
    sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    #cv2.subtract(img, laplacian, res, )

    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.show()

'''

#////////////////////////////
'''
傅里叶变化
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

flag = False

def getAddr(param):
    return os.getcwd() + '\\sourses\\' + param

if __name__ == '__main__':
    img = cv2.imread(getAddr('messigray.png'),0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    #构建振幅
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
'''