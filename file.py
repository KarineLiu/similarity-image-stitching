from sift import feature
import cv2 as cv
import os
import sys
from calibration import calibrate
import time

def img_rotation(mode, imgs):
    # 相机向右扫，不用翻转，
    # 相机向上扫描，图片向右翻转90度
    if mode == 2:
        for i in range(len(imgs)):
            tr = cv.transpose(imgs[i])
            imgs[i] = cv.flip(tr, 1)
    # 相机向左扫描，图片翻转180度
    elif mode == 3:
        for i in range(len(imgs)):
            imgs[i] = cv.flip(imgs[i], -1)
    # 相机向下扫描， 图片向左翻转90度
    elif mode == 4:
        for i in range(len(imgs)):
            tr = cv.transpose(imgs[i])
            imgs[i] = cv.flip(tr, 0)
    return imgs

def final_img(mode, result_img):
    if mode == 2:
        tr = cv.transpose(result_img)
        final = cv.flip(tr, 0)
    elif mode == 3:
        final = cv.flip(result_img, -1)
    elif mode == 4:
        tr = cv.transpose(result_img)
        final = cv.flip(tr, 1)
    else:
        final = result_img
    return final

def plane_projection_parameter(imga, imgb):
    h = imga.shape[0]
    w = imga.shape[1]
    template = imgb[0.4 * h: 0.6 * h, 0: 0.2 * w]
    res = cv.matchTemplate(imga, template, cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    dh = max_loc.y - 0.4 * h
    dw = max_loc.x
    return dh, dw

def plane_projection(dh, dw):

'''
/**柱面投影函数
 *参数列表中imgIn为输入图像，f为焦距
 *返回值为柱面投影后的图像
*/
Mat cylinder(Mat imgIn, int f)
{
    int colNum, rowNum;
    colNum = 2 * f*atan(0.5*imgIn.cols / f);//柱面图像宽
    rowNum = 0.5*imgIn.rows*f / sqrt(pow(f, 2)) + 0.5*imgIn.rows;//柱面图像高

    Mat imgOut = Mat::zeros(rowNum, colNum, CV_8UC1);
    Mat_<uchar> im1(imgIn);
    Mat_<uchar> im2(imgOut);

    //正向插值
    int x1(0), y1(0);
    for (int i = 0; i < imgIn.rows; i++)
        for (int j = 0; j < imgIn.cols; j++)
        {
            x1 = f*atan((j - 0.5*imgIn.cols) / f) + f*atan(0.5*imgIn.cols / f);
            y1 = f*(i - 0.5*imgIn.rows) / sqrt(pow(j - 0.5*imgIn.cols, 2) + pow(f, 2)) + 0.5*imgIn.rows;
            if (x1 >= 0 && x1 < colNum&&y1 >= 0 && y1<rowNum)
            {
                im2(y1, x1) = im1(i, j);
            }
        }
    return imgOut;
}
'''
if __name__ ==  '__main__':
    start = time.time()
    # -------------------------------------按帧读取图像-------------------------------------------------------
    path = r"D:\\trans\\data4\\"

    cap = cv.VideoCapture(path + "pip4.mp4")
    mode = 2

    i = 0
    j = 0
    fps = 45
    imgs = []

    success, frame = cap.read()

    imgs.append(frame)

    # ------------------------------------图像标定------------------------------------------------------------
    # 获取相机标定参数
    cal = calibrate()
    mtx, dist = cal.get_parameter()
    while success:
        i = i + 1
        if i % fps == 0:
            j = j + 1
            #cv.imwrite('before'+str(j)+'.jpg', frame)
            frame = cal.operate_calibration(mtx, dist, frame)
            #cv.imwrite('after'+str(j)+'.jpg', frame)
            imgs.append(frame)

        success, frame = cap.read()
    # -------------------------------根据镜头运动的主方向，进行图像旋转-------------------------------------------
    imgs = img_rotation(mode, imgs)


    # ---------------------------------图像预处理（平面重投影）------------------------------------------------------
    pics = []
    pics.append(imgs[0])
    for i in range(len(imgs) - 1):
        dh, dw = plane_projection_parameter(imgs[i], imgs[i + 1])
        pic = plane_projection(dh, dw)
        pics.append(pic)

    # ------------------------------------图像拼接---------------------------------------------------------------
    # 读取data下的全部文件
    fea = feature()
    final = imgs[0]
    for i in range(len(imgs) - 1):
        print(i)
        final = fea.stitching(final, imgs[i + 1], i, i + 1)
    final = final_img(mode, final)
    cv.imwrite("result.jpg", final)
    end = time.time()
    print("running time is %ds" % (end - start))
