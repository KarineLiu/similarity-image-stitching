import cv2 as cv
import os
import sys
import numpy as np

class calibrate():
    def get_parameter(self):
        path = r"D:\\calibration\\4k\\"
        file_name = os.listdir(path)
        imgs = []  # 存储彩色图像
        obj_points = []  # 存储3D点
        img_points = []  # 存储2D点

        # 初始化标定板角点的3D位置
        objp = np.zeros((8*11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
        i=0

        # todo 1 读入图片并完成角点检测
        for name in file_name:
            img = cv.imread(path+name)
            if img is None:
                print("can't read image"+name)
                sys.exit(-1)
            imgs.append(img)
            # ----------------------------------角点检测--------------------------------------
            gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            # size的长宽颠倒 从cv转到真实视图 输出为1920，1080
            size = gray.shape[::-1]
            # found表示找到内点状态 True/False, corners为角点坐标
            # 函数说明 https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html#gadc5bcb05cb21cf1e50963df26986d7c9
            '''ret, corners = cv.findChessboardCorners(gray, (11, 8), None)
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            if ret:

                obj_points.append(objp)

                corners2 = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
                # print(corners2)
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)

                cv.drawChessboardCorners(img, (11, 8), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
                i += 1
                cv.imwrite('conimg' + str(i) + '.jpg', img)
                
                '''
            found, corners = cv.findChessboardCornersSB(gray, (11, 8), cv.CALIB_CB_EXHAUSTIVE|cv.CALIB_CB_ACCURACY)

            if found:
                obj_points.append(objp)
                img_points.append(corners)
                # 画角点检测结果
                #dr = cv.drawChessboardCorners(img, (11, 8), corners, found)
                i += 1
                #cv.imwrite('conimg' + str(i) + '.jpg', img)

        # todo 2 标定
        # 函数说明https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
        # obj_points为世界坐标下的点
        # ret 总体 RMS 重投影误差; mtx 内参矩阵； dist 畸变系数； rvecs 旋转向量（外参）； tvecs 平移向量（外参）

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, size, None, None)
        print(ret)
        print(mtx)
        return mtx, dist

    def operate_calibration(self, mtx, dist, image):
        # todo 3 去畸变
        h, w = image.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # 显示更大范围的图片（正常重映射之后会删掉一部分图像）
        dst = cv.undistort(image, mtx, dist, None, newcameramtx)

        x, y, w, h = roi
        dst1 = dst[y:y + h, x:x + w]
        return dst1