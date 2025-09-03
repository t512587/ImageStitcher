import random

import numpy as np
import cv2
from matplotlib import pyplot as plt

from Blender import Blender
from Homography import Homography


class Stitcher:
    def __init__(self):
        pass

    def stitch(self, imgs, blending_mode="linearBlending", ratio=0.75):
        '''
            The main method to stitch image
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        print("Left img size (", hl, "*", wl, ")")
        print("Right img size (", hr, "*", wr, ")")

        # Step1 - extract the keypoints and features by SIFT detector and descriptor
        print("Step1 - Extract the keypoints and features by SIFT detector and descriptor...")
        kps_l, features_l = self.detectAndDescribe(img_left)
        kps_r, features_r = self.detectAndDescribe(img_right)

        # Step2 - extract the match point with threshold (David Lowe’s ratio test)
        print("Step2 - Extract the match point with threshold (David Lowe’s ratio test)...")
        matches_pos = self.matchKeyPoint(kps_l, kps_r, features_l, features_r, ratio)
        print("The number of matching points:", len(matches_pos))

        # Step2 - draw the img with matching point and their connection line
        self.drawMatches([img_left, img_right], matches_pos)

        # Step3 - fit the homography model with RANSAC algorithm
        print("Step3 - Fit the best homography model with RANSAC algorithm...")
        HomoMat = self.fitHomoMat(matches_pos)

        # Step4 - Warp image to create panoramic image
        print("Step4 - Warp image to create panoramic image...")
        warp_img = self.warp([img_left, img_right], HomoMat, blending_mode)

        return warp_img

    def detectAndDescribe(self, img):
        '''
        The Detector and Descriptor
        '''
        # SIFT detector and descriptor
        sift = cv2.SIFT_create()
        kps, features = sift.detectAndCompute(img, None)

        return kps, features



    def matchKeyPoint(self,kps_l, kps_r, features_l, features_r, ratio=0.75):
        """
        Match keypoints between two images using OpenCV BFMatcher with ratio test.
        Returns list of matched points: [[(x1, y1), (x2, y2)], ...]
        """
        # 建立 BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # SIFT descriptor 使用 L2 距離

        # KNN match 找到每個 descriptor 的前兩個最近鄰
        matches = bf.knnMatch(features_l, features_r, k=2)

        goodMatches_pos = []
        for m, n in matches:
            # Lowe ratio test
            if m.distance < ratio * n.distance:
                pt_left = kps_l[m.queryIdx].pt
                pt_right = kps_r[m.trainIdx].pt
                goodMatches_pos.append([(int(pt_left[0]), int(pt_left[1])),
                                        (int(pt_right[0]), int(pt_right[1]))])

        return goodMatches_pos


    def drawMatches(self, imgs, matches_pos):
        '''
            Draw the match points img with keypoints and connection line
        '''

        # initialize the output visualization image
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        vis = np.zeros((max(hl, hr), wl + wr, 3), dtype="uint8")
        vis[0:hl, 0:wl] = img_left
        vis[0:hr, wl:] = img_right

        # Draw the match
        for (img_left_pos, img_right_pos) in matches_pos:
            pos_l = img_left_pos
            pos_r = img_right_pos[0] + wl, img_right_pos[1]
            cv2.circle(vis, pos_l, 3, (0, 0, 255), 1)
            cv2.circle(vis, pos_r, 3, (0, 255, 0), 1)
            cv2.line(vis, pos_l, pos_r, (255, 0, 0), 1)

        # return the visualization
        plt.figure(4)
        plt.title("img with matching points")
        plt.imshow(vis[:, :, ::-1])
        # cv2.imwrite("Feature matching img/matching.jpg", vis)

        return vis

    def fitHomoMat(self, matches_pos):
        '''
            Fit the best homography model with RANSAC algorithm - noBlending、linearBlending、linearBlendingWithConstant
        '''
        # 轉成 NumPy array
        dstPoints = np.array([list(dst) for dst, src in matches_pos], dtype=np.float32)
        srcPoints = np.array([list(src) for dst, src in matches_pos], dtype=np.float32)

        # OpenCV 計算 Homography，使用 RANSAC
        H, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, ransacReprojThreshold=5.0)

        # mask 是 inlier 的標記，可以用來統計
        num_inliers = np.sum(mask)
        print("The Number of Maximum Inlier:", num_inliers)

        return H


    def warp(self, imgs, HomoMat, blending_mode):
        '''
           Warp image to create panoramic image
           There are three different blending method - noBlending、linearBlending、linearBlendingWithConstant
        '''
        img_left, img_right = imgs
        hl, wl = img_left.shape[:2]
        hr, wr = img_right.shape[:2]

        # 建立拼接圖的大小
        stitch_width = wl + wr
        stitch_height = max(hl, hr)
        stitch_img = np.zeros((stitch_height, stitch_width, 3), dtype=np.uint8)

        # 把左圖直接放進去
        stitch_img[:hl, :wl] = img_left

        # Warp 右圖
        warped_right = cv2.warpPerspective(
            img_right,
            HomoMat,
            (stitch_width, stitch_height),  # 輸出尺寸
            flags=cv2.INTER_LINEAR  # 線性插值，可以改成 INTER_NEAREST 或 INTER_CUBIC
        )
        if blending_mode == "noBlending":
            # 直接覆蓋右圖非零區域
            mask = (warped_right > 0)
            stitch_img[mask] = warped_right[mask]
        # remove the black border
        stitch_img = self.removeBlackBorder(stitch_img)

        return stitch_img

    def removeBlackBorder(self, img):
        '''
        Remove img's the black border
        '''
        h, w = img.shape[:2]
        reduced_h, reduced_w = h, w
        # right to left
        for col in range(w - 1, -1, -1):
            all_black = True
            for i in range(h):
                if (np.count_nonzero(img[i, col]) > 0):
                    all_black = False
                    break
            if (all_black == True):
                reduced_w = reduced_w - 1

        # bottom to top
        for row in range(h - 1, -1, -1):
            all_black = True
            for i in range(reduced_w):
                if (np.count_nonzero(img[row, i]) > 0):
                    all_black = False
                    break
            if (all_black == True):
                reduced_h = reduced_h - 1

        return img[:reduced_h, :reduced_w]
