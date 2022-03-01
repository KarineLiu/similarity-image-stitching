import cv2 as cv
import numpy as np
import math
import time


# 定义图形拼接类
class feature():
    # 定义图形拼接类
    def stitching(self, imageB, imageA, nameB, nameA):
        # TODO 1 特征点提取
        kps1, fea1 = self.siftdetect(nameA, imageA)
        kps2, fea2 = self.siftdetect(nameB, imageB)
        # TODO 2 特征点匹配
        # pA pB为有效特征匹配对pa(x,y)而opencv是（y,x）
        (pA, pB, matches, h, status) = self.matchKeypoints(kps1, kps2, fea1, fea2)
        # TODO 3 计算相似变换矩阵
        S = self.computeHomography(pA, pB)
        # TODO 4 图像融合
        # 不考虑图像融合, 简单的扭曲拼接
        result_img = self.global_fusion(imageA, imageB, nameB, S)
        #result_img = self.fusion(imgA2, imgB2, nameB, trans_result, h)
        # self.drawMatches(imageA, imageB, pA, pB, nameB)
        return result_img

    # TODO 1 特征点提取
    # 调用SIFT.create()生成SIFT特征 i区分不同的图片
    def siftdetect(self, i, image):
        # 将彩色图片转换成灰度图, 灰度匹配信息减少失效？
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 建立SIFT生成器
        sift = cv.SIFT_create()
        # 检测SIFT特征点，并计算描述子,kps表示关键点
        (kps, features) = sift.detectAndCompute(gray, None)
        # 将结果kps原本的list转换成NumPy数组 kp[i].pt.x就是横坐标
        # 这步后是[[x1,y1][]...]这样的数组
        kps = np.float32([kp.pt for kp in kps])
        # 返回特征点集，及对应的描述特征
        return (kps, features)

    # TODO 2 特征点匹配
    # 返回相应的匹配结果和单应变矩阵
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, lowe_ratio=0.75, reprojThresh=2):
        # 建立原始暴力匹配器，应用opencv提供的欧式距离，寻找所有匹配的可能，从小到大排序
        matcher = cv.BFMatcher()
        # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        # 进一步的精确点儿的匹配
        matches = []
        for m in rawMatches:
            # 存在两个最近邻当最近距离跟次近距离的比值小于lowe_ratio值时，保留此匹配对
            if len(m) == 2 and m[0].distance < m[1].distance * lowe_ratio:
                # 存储两个点在featuresA, featuresB中的索引值
                # Dmatch中的trainIdx和queryIdx即为匹配结果
                matches.append((m[0].trainIdx, m[0].queryIdx))
        # 当筛选后的匹配对大于4时，计算视角变换矩阵
        if len(matches) > 4:
            # 获取匹配对的点坐标，找另外一个图上对应的坐标，一次性完成多组匹配的坐标？
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # 计算视角变换矩阵，在这里是单应变矩阵，用到了RANSAC
            # 0 - 利用所有点的常规方法; RANSAC - RANSAC-基于RANSAC的鲁棒算法
            # LMEDS - 最小中值鲁棒算法; RHO - PROSAC-基于PROSAC的鲁棒算法
            # RANSAC（随机采样一致性）：该方法利用匹配点计算两个图像之间单应矩阵，然后利用重投影误差来判定某一个匹配是不是正确的匹配
            # reprojThresh 为一个表征阈值的量1-10, 将点对视为内点的最大允许重投影错误阈值
            (h, status) = cv.findHomography(ptsA, ptsB, cv.RANSAC, reprojThresh)
            label = []
            pA = []
            pB = []
            for i in range(len(status)):
                if status[i] > 0:
                    label.append(i)
                    pA.append(ptsA[i])
                    pB.append(ptsB[i])
            # findHomography (A, B, method, thresh)
            # 返回结果
            return (pA, pB, matches, h, status)
        # 如果匹配对小于4时，返回None
        return None

    # 单独引入尺度变换
    def computeHomography(self, pa, pb):
        num = len(pa)
        A = np.zeros((2 * num, 3))
        b = np.zeros((2 * num, 1))
        T = np.zeros((3, 3))
        for i in range(num):
            A[2 * i][0] = pa[i][0]
            A[2 * i][1] = 1
            A[2 * i][2] = 0
            A[2 * i + 1][0] = pa[i][1]
            A[2 * i + 1][1] = 0
            A[2 * i + 1][2] = 1
            b[2 * i][0] = pb[i][0]
            b[2 * i + 1][0] = pb[i][1]
        U, sigma, Vt = np.linalg.svd(A)
        s = np.diag(sigma)
        # s为分解出来的奇异值，一维矩阵
        un = U[:, :3]
        xx = np.dot(s, Vt)
        x = np.dot(np.linalg.inv(xx), np.transpose(un))
        x = np.dot(x, b)
        T[0][0] = x[0][0]
        T[0][2] = x[1][0]
        T[1][2] = x[2][0]
        T[1][1] = x[0][0]
        T[2][2] = 1

        return(T)

    '''
    # TODO 3 计算相似变换矩阵S 四个自由度
    # 求解问题http://math.ecnu.edu.cn/~jypan/Teaching/NA/slides_03D_LS.pdf
    def computeHomography(self, pa, pb):
        #     [alpha, -beta, t1]
        # H = [beta, alpha, t2]
        #     [0, 0, 1]
        num = len(pa)
        # b=Ax, x=[alpha,beta,t1,t2]T,
        # b=[x1'，y1',x2',y2',...]T,A--[[x1, -y1,1,0],[y1,x1,0,1],...]
        # 构造矩阵A(2n*4)和b(2n*1)
        A = np.zeros((2 * num, 4))
        b = np.zeros((2 * num, 1))
        # 构造相似矩阵
        S = np.zeros((3, 3))
        # 计算方法 QR矩阵分解+最小二乘法 https://www.cnblogs.com/caimagic/p/12202884.html
        for i in range(num):
            A[2 * i][0] = pa[i][0]
            A[2 * i][1] = -pa[i][1]
            A[2 * i][2] = 1
            A[2 * i][3] = 0
            A[2 * i + 1][0] = pa[i][1]
            A[2 * i + 1][1] = pa[i][0]
            A[2 * i + 1][2] = 0
            A[2 * i + 1][3] = 1
            b[2 * i][0] = pb[i][0]
            b[2 * i + 1][0] = pb[i][1]


        # --------------------------------qr算法---------------------------------
        #q, r = np.linalg.qr(A)
        #x = np.dot(np.linalg.inv(r), np.transpose(q))
        #x = np.dot(x, b)
        # -------------------------------直接求解法--------------------------------
        #xx1 = np.dot(np.transpose(A), A)
        #x = np.dot(np.linalg.inv(xx1), np.transpose(A))
        #x = np.dot(x, b)
        # -----------------------------SVD算法-----------------------------------
        U, sigma, Vt = np.linalg.svd(A)
        s = np.diag(sigma)
        # s为分解出来的奇异值，一维矩阵
        un = U[:, :4]
        xx = np.dot(s, Vt)
        x = np.dot(np.linalg.inv(xx), np.transpose(un))
        x = np.dot(x, b)

        S[0][0] = x[0][0]
        S[0][1] = -x[1][0]
        S[0][2] = x[2][0]
        S[1][0] = x[1][0]
        S[1][1] = x[0][0]
        S[1][2] = x[3][0]
        S[2][2] = 1

        return(S)'''

    # todo 4.2 计算H矩阵作用后的图片
    def warp(self, imageA, H, maxx, maxy):
        # image.shape[0], 图片垂直尺寸h
        # image.shape[1], 图片水平尺寸w
        # image.shape[2], 图片通道数
        # imageA为待处理图像 H为扭曲矩阵，输出图片大小
        warp_image = cv.warpPerspective(imageA, H, (maxx, maxy))
        return warp_image

    # todo 4.1 计算边界
    # 计算H扭曲后的新输入的图像边界(minx miny maxx maxy)
    def boundingbox(self, img, H):
        # input : img & H
        # output : minxy maxxy

        # 定义一个舍入函数
        def roundup(x):
            if x < 0.0:
                x = math.floor(x)
            else:
                x = math.ceil(x)
            return x

        h = img.shape[0] # 图片高度
        w = img.shape[1] # 图片宽度

        # [row, line, normalization_vector]
        upper_left = np.array([0, 0, 1])
        lower_left = np.array([0, h-1, 1])
        upper_right = np.array([w-1, 0, 1])
        lower_right = np.array([w-1, h-1, 1])

        # 与相应的系数相乘
        upper_left = np.dot(H, upper_left)
        lower_left = np.dot(H, lower_left)
        upper_right = np.dot(H, upper_right)
        lower_right = np.dot(H, lower_right)

        # 计算min x、y max x、y
        minx = roundup(min(upper_left[0]/upper_left[2], lower_left[0]/lower_left[2],
                           upper_right[0]/upper_right[2], lower_right[0]/lower_right[2]))
        maxx = roundup(max(upper_left[0] / upper_left[2], lower_left[0] / lower_left[2],
                           upper_right[0] / upper_right[2], lower_right[0] / lower_right[2]))
        miny = roundup(min(upper_left[1] / upper_left[2], lower_left[1] / lower_left[2],
                           upper_right[1] / upper_right[2], lower_right[1] / lower_right[2]))
        maxy = roundup(max(upper_left[1] / upper_left[2], lower_left[1] / lower_left[2],
                           upper_right[1] / upper_right[2], lower_right[1] / lower_right[2]))
        return int(minx), int(miny), int(maxx), int(maxy)

    # todo 4.3 计算最佳缝合线各像素点能量
    def compute_energy(self, ma, mb):
        # --------------色调约束-----------------
        hsv_ma = cv.cvtColor(ma, cv.COLOR_BGR2HSV)
        hsv_mb = cv.cvtColor(mb, cv.COLOR_BGR2HSV)
        # --------------形状约束-----------------
        gray_ma = cv.cvtColor(ma, cv.COLOR_RGB2GRAY)
        gray_mb = cv.cvtColor(mb, cv.COLOR_RGB2GRAY)
        # --------------Sobel算子---------------
        Sx = np.array([[-2, 0, 2], [-1, 0, 1], [-2, 0, 2]])
        Sy = np.array([[-2, -1, -2], [0, 0, 0], [2, 1, 2]])
        # --------------边缘检测-----------------
        ma_Sx = cv.filter2D(gray_ma, -1, Sx)
        ma_Sy = cv.filter2D(gray_ma, -1, Sy)
        mb_Sx = cv.filter2D(gray_mb, -1, Sx)
        mb_Sy = cv.filter2D(gray_mb, -1, Sy)
        # ------------一个区域化的矩阵------------
        F = np.array([[1, 1, 1], [1, 8, 1], [1, 1, 1]])
        # 结构差异
        E_geometry = (ma_Sx - mb_Sx) ** 2 + (ma_Sy - mb_Sy) ** 2
        # 灰度/强度差异(表征邻域的灰度差异)
        E_intensity = (gray_ma-gray_mb) ** 2
        E_intensity = cv.filter2D(E_intensity, -1, F)
        # 颜色差异
        E_color = abs(hsv_ma[:, :, 0] - hsv_mb[:, :, 0])
        E = E_color + E_intensity + E_geometry
        E2 = np.zeros((ma.shape[0], ma.shape[1]))
        # 对黑色区域进行惩罚
        for i in range(ma.shape[0]):
            for j in range(ma.shape[1]):
                if gray_ma[i,j] == 0 or gray_mb[i, j] == 0:
                    E2[i, j] = E[i, j] + 5000
                else:
                    E2[i, j] = E[i, j]
        return E2

    # todo 4.3 计算最佳缝合线
    def stitch_line(self, width, imga, imgb, minx, maxx, miny, maxy):
        imga2 = imga[miny: maxy, minx:maxx]
        imgb2 = imgb[miny: maxy, minx:maxx]
        E = self.compute_energy(imga2, imgb2)
        # 记录位置索引
        locationy = np.zeros((imga2.shape[0], imga2.shape[1])).astype(int)
        # 记录动态规划的值
        accumulate = np.zeros((imga2.shape[0], imga2.shape[1])).astype(int)
        # 先初始化上左右的值
        # 上边框
        for j in range(imga2.shape[1]):
            accumulate[0, j] = E[0, j]
            locationy[0, j] = j
        # 遍历每一行
        for i in range(1, imga2.shape[0]):
            # 遍历该行的每一列赋值
            for j in range(imga2.shape[1]):
                # 先赋初值
                acc_min = accumulate[i-1, j]
                locationy[i, j] = j
                # 遍历上一行的width限定的能量传递范围的每一列
                for k in range(max(0, j-width), min(imga2.shape[1]-1, j+width)+1):
                    if accumulate[i - 1, k] < acc_min:
                        acc_min = accumulate[i - 1, k]
                        locationy[i, j] = k
                accumulate[i, j] = acc_min + E[i, j]
        # ------------------------------------回溯-------------------------------------
        min_accumulate = accumulate[imga2.shape[0]-1, 0]
        lo = 0
        for j in range(1, imga2.shape[1]):
            if accumulate[imga2.shape[0]-1, j] < min_accumulate:
                min_accumulate = accumulate[imga2.shape[0]-1, j]
                lo = j
        ly = np.zeros((imga2.shape[0], 1)).astype(int)
        ly[-1] = lo
        for i in range(imga2.shape[0]-2, -1, -1):
            # lo 为第i行对应切割点的列坐标，即第i行该分开的地方
            lo = locationy[i+1, lo]
            ly[i] = lo
        return ly

    # TODO 4 图像拼贴
    def global_fusion(self, imageA, imageB, nameA, H):
      result_path = "D:\\trans\\result\\"
      # todo 4.1 计算边界
      # Bounding_box 表示imga经过矩阵H warp后图像的边界
      minx, miny, maxx, maxy = self.boundingbox(imageA, H)
      minx = max(0, minx)
      miny = max(0, miny)
      # 计算最终图片的尺寸
      row = max(imageB.shape[1], maxx)
      line = max(imageB.shape[0], maxy)
      cross_minx = row
      cross_miny = line
      cross_maxx = 0
      cross_maxy = 0
      # 生成2个空白图片
      result_img = np.zeros((line, row, 3), np.uint8)
      # 生成一个空的掩膜
      mask = np.zeros((line, row), np.uint8)
      # 先填入背景
      result_img[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
      # todo 4.2 计算扭曲图片
      warp_img = self.warp(imageA, H, maxx, maxy)
      # 转为灰度图像，计算掩膜
      gray_warp_img = cv.cvtColor(warp_img, cv.COLOR_RGB2GRAY)
      gray_background = cv.cvtColor(result_img, cv.COLOR_RGB2GRAY)
      # 计算重叠区域的掩膜和坐标范围
      for i in range(miny, maxy, 1):
          for j in range(minx, maxx, 1):
              if gray_warp_img[i, j] > 0 and gray_background[i, j] > 0:
                  cross_miny = min(cross_miny, i)
                  cross_maxy = max(cross_maxy, i)
                  cross_minx = min(cross_minx, j)
                  cross_maxx = max(cross_maxx, j)
                  mask[i, j] = 255
      #mask_warp = cv. bitwise_and(warp_img, warp_img, mask=mask)
      #mask_result = cv.bitwise_and(result_img, result_img, mask=mask)
      # todo 4.3 最佳缝合线
      ly = self.stitch_line(1, warp_img, result_img, cross_minx, cross_maxx, cross_miny, cross_maxy)
      # 基于最佳接缝线的普通融合
      for i in range(cross_miny, cross_maxy):
          # 计算得到的接缝线坐标为beam_point
          beam_point = int(cross_minx + int(ly[i - cross_miny]))
          # 画出接缝线
          cv.circle(result_img, (beam_point, i), 5, (0, 140, 255), -1)
          for j in range(beam_point, cross_maxx):
              if warp_img[i, j, 0] > 0 or warp_img[i, j, 1] > 0 or warp_img[i, j, 2] > 0:
                  result_img[i, j] = warp_img[i, j]
                  #result_img[i, j] = warp_img[i-cross_miny, j-cross_minx]
          if row == imageA.shape[1]:
              for j in range(cross_maxx, maxx):
                  result_img[i, j] = warp_img[i, j]
      # 填充A的区域

      for i in range(maxy):
          for j in range(maxx):
              if result_img[i, j, 0] == 0 and result_img[i, j, 1] == 0 and result_img[i, j, 0] == 0:
                  if warp_img[i, j, 0] > 0 or warp_img[i, j, 1] > 0 or warp_img[i, j, 2] > 0:
                      result_img[i, j] = warp_img[i, j]

      cv.imwrite(result_path + "panorama%d.jpg" % (nameA), result_img)
      return result_img


'''
# 最初的单平移拼接版本
# 定义图形拼接类
class feature():
    # 整合图像拼接过程
    def stitching(self, imageB, imageA, nameB, nameA):
        # 特征点提取
        kps1, fea1 = self.siftdetect(nameA, imageA)
        kps2, fea2 = self.siftdetect(nameB, imageB)
        # 特征点匹配
        # pA pB为有效特征匹配对pa(x,y)而opencv是（y,x）
        (pA, pB, matches, h, status) = self.matchKeypoints(kps1, kps2, fea1, fea2)
        # 先完成平移
        img_trans = tran(imageA, imageB, np.array(pA), np.array(pB), nameA)
        (imgA2, imgB2, trans_result) = img_trans.justtrans()
        # 不考虑图像融合, 简单的扭曲拼接
        result_img = self.fusion(imgA2, imgB2, nameB,trans_result, h)
        #self.drawMatches(imageA, imageB, pA, pB, nameB)
        return result_img

    # 画出图片的匹配结果
    def drawMatches(self, img1, img2, kpsA, kpsB, name):
        # 初始化可视化图片，将A、B图左右连接到一起
        # 计算初始图像大小（高，宽）
        (hA, wA) = img1.shape[:2]
        (hB, wB) = img2.shape[:2]
        # 平铺排列两张图
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hB, 0:wB] = img2
        vis[0:hA, wB:] = img1
        # 联合遍历，画出匹配对
        for i in range(len(kpsA)):
            ptA = (int(kpsA[i][0])+wB, int(kpsA[i][1]))
            ptB = (int(kpsB[i][0]), int(kpsB[i][1]))
            cv.line(vis, ptA, ptB, (0, 255, 255), 2)
        path = "F:\\cv_ex\\trans\\stitch\\"
        cv.imwrite(path + "stitch%d.jpg" % (name), vis)
        # cv.waitKey(0)
        # return可视化结果
        return 0

    # 单纯的2个图像拼接
    def fusion(self, imageA, imageB, nameA, trans, H):
      # imgA和B为平移后的图片
      result_path = "F:\\cv_ex\\trans\\result\\"
      row = max(imageA.shape[1], imageB.shape[1])
      line = max(imageA.shape[0], imageB.shape[0])
      print(imageB.shape)
      print(imageA.shape)
      # 生成一个空白图片
      result_img = np.zeros((line, row, 3), np.uint8)
      print(result_img.shape)
      # 先填入背景
      result_img[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
      #--------------------------计算掩膜---------------------------------------
      # 转为灰度图像，计算掩膜
      gray_imgA= cv.cvtColor(imageA, cv.COLOR_RGB2GRAY)
      gray_background = cv.cvtColor(result_img, cv.COLOR_RGB2GRAY)
      cross_maxx = 0
      cross_minx = row
      cross_maxy = 0
      cross_miny = line
      # 计算重叠区域
      for i in range(imageA.shape[0]):
          for j in range(imageA.shape[1]):
              if gray_imgA[i, j] > 0 and gray_background[i, j] > 0:
                  cross_minx = min(cross_minx, j)
                  cross_maxx = max(cross_maxx, j)
                  cross_miny = min(cross_miny, i)
                  cross_maxy = max(cross_maxy, i)
      # -------------------------基于最佳缝合线完成拼接----------------------
      ly = self.stitch_line(3, imageA[cross_miny:cross_maxy,cross_minx:cross_maxx], imageB[cross_miny:cross_maxy,cross_minx:cross_maxx], cross_minx ,cross_maxx, cross_miny, cross_maxy)
      # 基于最佳接缝线的普通融合
      # 重叠区域的处理
      for i in range(cross_miny, cross_maxy):
          # 计算得到的接缝线坐标为beam_point
          beam_point = int(cross_minx + int(ly[i - cross_miny]))
          # 画出接缝线
          # cv.circle(result_img, (beam_point, i), 5, (0, 140, 255), -1)
          for j in range(beam_point, cross_maxx):
              if imageA[i, j, 0] > 0 or imageA[i, j, 1] > 0 or imageA[i, j, 2] > 0:
                  result_img[i, j] = imageA[i, j]
          if row == imageA.shape[1]:
              for j in range(cross_maxx, row):
                  result_img[i, j] = imageA[i, j]

      for i in range(imageA.shape[0]):
          for j in range(imageA.shape[1]):
              if result_img[i, j, 0] == 0 and result_img[i, j, 1] == 0 and result_img[i, j, 0] == 0:
                  if imageA[i, j, 0] > 0 or imageA[i, j, 1] > 0 or imageA[i, j, 2] > 0:
                      result_img[i, j] = imageA[i, j]
      cv.imwrite(result_path + "panorama%d.jpg" % (nameA), result_img)
      return result_img


    def compute_energy(self, ma, mb):
        # --------------色调约束-----------------
        hsv_ma = cv.cvtColor(ma, cv.COLOR_BGR2HSV)
        hsv_mb = cv.cvtColor(mb, cv.COLOR_BGR2HSV)
        # --------------形状约束-----------------
        gray_ma = cv.cvtColor(ma, cv.COLOR_RGB2GRAY)
        gray_mb = cv.cvtColor(mb, cv.COLOR_RGB2GRAY)
        # --------------Sobel算子---------------
        Sx = np.array([[-2, 0, 2], [-1, 0, 1], [-2, 0, 2]])
        Sy = np.array([[-2, -1, -2], [0, 0, 0], [2, 1, 2]])
        # --------------边缘检测-----------------
        ma_Sx = cv.filter2D(gray_ma, -1, Sx)
        ma_Sy = cv.filter2D(gray_ma, -1, Sy)
        mb_Sx = cv.filter2D(gray_mb, -1, Sx)
        mb_Sy = cv.filter2D(gray_mb, -1, Sy)
        # ------------一个区域化的矩阵------------
        F = np.array([[1, 1, 1], [1, 8, 1], [1, 1, 1]])
        # 结构差异
        E_geometry = (ma_Sx - mb_Sx) ** 2 + (ma_Sy - mb_Sy) ** 2
        # 灰度/强度差异(表征邻域的灰度差异)
        E_intensity = (gray_ma-gray_mb) ** 2
        E_intensity = cv.filter2D(E_intensity, -1, F)
        # 颜色差异
        E_color = (hsv_ma[:, :, 0] - hsv_mb[:, :, 0])
        E = E_color + E_intensity + E_geometry
        return E

    # 图像融合
    # 计算H扭曲后的新输入的图像边界(minx miny maxx maxy)
    # 计算最佳缝合线
    def stitch_line(self, width, imga, imgb, minx, maxx, miny, maxy):
        #imga2 = imga[miny: maxy, minx:maxx]
        #imgb2 = imgb[miny: maxy, minx:maxx]
        imga2 = imga
        imgb2 = imgb
        E = self.compute_energy(imga2, imgb2)
        # 记录位置索引
        locationy = np.zeros((imga2.shape[0], imga2.shape[1])).astype(int)
        # 记录动态规划的值
        accumulate = np.zeros((imga2.shape[0], imga2.shape[1])).astype(int)
        # 先初始化上左右的值
        # 上边框
        for j in range(imga2.shape[1]):
            accumulate[0, j] = E[0, j]
            locationy[0, j] = j
        # 遍历每一行
        for i in range(1, imga2.shape[0]):
            # 遍历该行的每一列赋值
            for j in range(imga2.shape[1]):
                # 先赋初值
                acc_min = accumulate[i-1, j]
                locationy[i, j] = j
                # 遍历上一行的width限定的能量传递范围的每一列
                for k in range(max(0, j-width), min(imga2.shape[1]-1, j+width)+1):
                    if accumulate[i - 1, k] < acc_min:
                        acc_min = accumulate[i - 1, k]
                        locationy[i, j] = k
                accumulate[i, j] = acc_min + E[i, j]
        # ------------------------------------回溯-------------------------------------
        min_accumulate = accumulate[imga2.shape[0]-1, 0]
        lo = 0
        for j in range(1, imga2.shape[1]):
            if accumulate[imga2.shape[0]-1, j] < min_accumulate:
                min_accumulate = accumulate[imga2.shape[0]-1, j]
                lo = j
        ly = np.zeros((imga2.shape[0], 1)).astype(int)
        ly[-1] = lo
        for i in range(imga2.shape[0]-2, -1, -1):
            # lo 为第i行对应切割点的列坐标，即第i行该分开的地方
            lo = locationy[i+1, lo]
            ly[i] = lo
        return ly


    # 图像融合
    # 调用SIFT.create()生成SIFT特征 i区分不同的图片
    def siftdetect(self, i, image):
        # 将彩色图片转换成灰度图, 灰度匹配信息减少失效？
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 建立SIFT生成器
        sift = cv.SIFT_create()
        # 检测SIFT特征点，并计算描述子,kps表示关键点
        (kps, features) = sift.detectAndCompute(gray, None)

        # 将结果kps原本的list转换成NumPy数组 kp[i].pt.x就是横坐标
        # 这步后是[[x1,y1][]...]这样的数组
        kps = np.float32([kp.pt for kp in kps])
        # 返回特征点集，及对应的描述特征
        return (kps, features)


    # 特征点匹配 返回相应的匹配结果和单应变矩阵
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, lowe_ratio = 0.75, reprojThresh = 3):
        # 建立原始暴力匹配器，应用opencv提供的欧式距离，寻找所有匹配的可能，从小到大排序
        matcher = cv.BFMatcher()
        # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        # 进一步的精确点儿的匹配
        matches = []
        for m in rawMatches:
            # 存在两个最近邻当最近距离跟次近距离的比值小于lowe_ratio值时，保留此匹配对
            if len(m) == 2 and m[0].distance < m[1].distance * lowe_ratio:
                # 存储两个点在featuresA, featuresB中的索引值
                # Dmatch中的trainIdx和queryIdx即为匹配结果
                matches.append((m[0].trainIdx, m[0].queryIdx))
            # 当筛选后的匹配对大于4时，计算视角变换矩阵
        if len(matches) > 4:
            # 获取匹配对的点坐标，找另外一个图上对应的坐标，一次性完成多组匹配的坐标？
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # 计算视角变换矩阵，在这里是单应变矩阵，用到了RANSAC
            # 0 - 利用所有点的常规方法; RANSAC - RANSAC-基于RANSAC的鲁棒算法
            # LMEDS - 最小中值鲁棒算法; RHO - PROSAC-基于PROSAC的鲁棒算法
            # RANSAC（随机采样一致性）：该方法利用匹配点计算两个图像之间单应矩阵，然后利用重投影误差来判定某一个匹配是不是正确的匹配
            # reprojThresh 为一个表征阈值的量1-10, 将点对视为内点的最大允许重投影错误阈值
            (h, status) = cv.findHomography(ptsA, ptsB, cv.RANSAC, reprojThresh)
            label = []
            pA = []
            pB = []
            for i in range(len(status)):
                if status[i] > 0:
                    label.append(i)
                    pA.append(ptsA[i])
                    pB.append(ptsB[i])
            # findHomography (A, B, method, thresh)
            # 返回结果
            return (pA, pB, matches, h, status)
        # 如果匹配对小于4时，返回None
        return None

    # 图像融合
    # 计算H矩阵作用后的图片
    def warp(self, imageA, H, maxx, maxy):
        # image.shape[0], 图片垂直尺寸h
        # image.shape[1], 图片水平尺寸w
        # image.shape[2], 图片通道数
        # imageA为待处理图像 H为扭曲矩阵，输出图片大小
        warp_image = cv.warpPerspective(imageA, H, (maxx, maxy))
        return warp_image

    # 画出图片的匹配结果
    def drawMatches(self, img1, img2, kpsA, kpsB, name):
        # 初始化可视化图片，将A、B图左右连接到一起
        # 计算初始图像大小（高，宽）
        (hA, wA) = img1.shape[:2]
        (hB, wB) = img2.shape[:2]
        # 平铺排列两张图
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hB, 0:wB] = img2
        vis[0:hA, wB:] = img1
        # 联合遍历，画出匹配对
        for i in range(len(kpsA)):
            ptA = (int(kpsA[i][0])+wB, int(kpsA[i][1]))
            ptB = (int(kpsB[i][0]), int(kpsB[i][1]))
            cv.line(vis, ptA, ptB, (0, 255, 255), 2)
        path = "F:\\cv_ex\\trans\\stitch\\"
        cv.imwrite(path + "stitch%d.jpg" % (name), vis)
        # cv.waitKey(0)
        # return可视化结果
        return 0
'''