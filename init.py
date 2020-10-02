import cv2
import numpy as np
import os
import myfunc as mf

sudoku_map = [[0]*mf.SUDOKU_SIZE for i in range(mf.SUDOKU_SIZE)]
sudoku_vec = []
sudoku_img = []

if __name__ == '__main__':
    print('请在相同文件夹下新建一个output文件夹，用来存放识别出的数字')
    addr = input('输入想要识别的数独图片名(带后缀):')
    #addr = 'sudoku1.png'
    #addr = 'drawing.png'
    #addr = 'sudoku_net.jpg'

    recimg = mf.RecImg(addr)
    oriimg = mf.RecImg(addr)
    #print(recimg.high, recimg.width, recimg.color_num)

    recimg.transGrid()
    recimg.binGrid(15,9)
    #recimg.show('binary original graph')
    recimg.medianBlur(3)
    #recimg.show('blur original graph') 


    #该函数的作用是在二值函数中寻找轮廓
    img,contours,hierachy = cv2.findContours(recimg.sudoku, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = -1
    for i in range(len(contours)):
        max_rec_tmp = cv2.approxPolyDP(contours[i], 4, True)
        
        max_area_tmp = cv2.contourArea(max_rec_tmp)
        if max_area_tmp > max_area:
            max_area = max_area_tmp
            max_rec = max_rec_tmp   
    max_rec_fin = max_rec

    #确定边上边角的四个点
    bungle_points = mf.countBunglePoint(max_rec_fin)
    #画点确定位置
    mf.testPoints(oriimg, bungle_points, 'testPoints')

    dst_points = [[0,0],
                [0,mf.FIN_SIZE*mf.SUDOKU_SIZE-1],
                [mf.FIN_SIZE*mf.SUDOKU_SIZE-1,0],
                [mf.FIN_SIZE*mf.SUDOKU_SIZE-1,mf.FIN_SIZE*mf.SUDOKU_SIZE-1]]

    #最初照片的边缘四个点
    bungle_points = np.float32(bungle_points)
    #变换后的点
    dst_points = np.float32(dst_points)
    mat = cv2.getPerspectiveTransform(bungle_points, dst_points)
    dst = cv2.warpPerspective(oriimg.sudoku, mat, (mf.FIN_SIZE*mf.SUDOKU_SIZE,mf.FIN_SIZE*mf.SUDOKU_SIZE))
    mf.testPoints(oriimg, bungle_points, 'testPoints')

    recimg.updateImg(dst)
    recimg.show('sudoku block')

    number_num = 0
    for i in range(mf.SUDOKU_SIZE):
        for j in range(mf.SUDOKU_SIZE):
            if mf.recognizedNum(i,j,recimg.sudoku, sudoku_img) == 0:
                continue
            sudoku_map[i][j] = 1
            sudoku_vec.append([i,j])
            cv2.imwrite(mf.getOutAddr('numrect x-'+str(i)+' y-'+str(j)+'.jpg'), sudoku_img[number_num])
            number_num = number_num + 1


    cv2.waitKey(0)
    cv2.destroyAllWindows()