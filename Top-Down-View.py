import cv2
import numpy as np
from Blender import Blender  # 確保你有這個模組

class ImageStitcher:
    def __init__(self):
        pass

    def removeBlackBorder(self, img):
        """快速移除拼接後的黑邊"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)
        if coords is None:  # 全黑避免錯誤
            return img
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]

    def _matrix_square_root(self, A):
        """計算 3x3 矩陣平方根"""
        try:
            vals, vecs = np.linalg.eig(A)
            sqrt_vals = np.sqrt(vals.astype(complex))  # 確保可以處理負數特徵值
            V = vecs
            Vinv = np.linalg.inv(V)
            M = V.dot(np.diag(sqrt_vals)).dot(Vinv)
            if np.max(np.abs(M.imag)) < 1e-6:
                M = M.real
            else:
                print("⚠️ 矩陣平方根包含虛數部分，取實部近似")
                M = M.real  # 取實部近似
            return M
        except Exception as e:
            print(f"⚠️ 計算矩陣平方根時發生錯誤: {e}")
            return None

    def _transform_points(self, H, pts):
        """使用 Homography 變換點集"""
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts_trans = (H @ pts_h.T).T
        # 避免除以零
        mask = np.abs(pts_trans[:, 2]) > 1e-8
        pts_trans[mask] /= pts_trans[mask, 2][:, np.newaxis]
        return pts_trans[:, :2]

    def warp(self, imgs, HomoMat):
        """拼接影像"""
        img_left, img_right = imgs

        try:
            blender = Blender()
        except ImportError:
            print("⚠️ 無法導入 Blender 模組，使用簡單混合")
            blender = None

        # 確保 Homography 是 3x3
        H = HomoMat.astype(np.float64)
        
        # 確保 H[2,2] 不為零
        if abs(H[2,2]) < 1e-8:
            print("⚠️ Homography 矩陣異常")
            return None

        # 計算平方根矩陣
        M = self._matrix_square_root(H)
        if M is None:
            print("⚠️ 無法計算 Homography 平方根，使用原始矩陣")
            # fallback: 使用簡單的拼接方法
            return self._simple_stitch(img_left, img_right, H)

        # 計算逆矩陣
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            print("⚠️ 無法計算逆矩陣")
            return None

        # Normalize
        M = M / M[2,2]
        Minv = Minv / Minv[2,2]

        # 計算邊界
        hl, wl = img_left.shape[:2]
        hr, wr = img_right.shape[:2]

        corners_left = np.array([[0,0],[wl,0],[wl,hl],[0,hl]], dtype=np.float64)
        corners_right = np.array([[0,0],[wr,0],[wr,hr],[0,hr]], dtype=np.float64)

        tl = self._transform_points(Minv, corners_left)
        tr = self._transform_points(M, corners_right)

        all_pts = np.vstack([tl, tr])
        min_x, min_y = np.min(all_pts, axis=0)
        max_x, max_y = np.max(all_pts, axis=0)

        tx, ty = -min_x, -min_y
        T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float64)

        out_w = int(np.ceil(max_x - min_x))
        out_h = int(np.ceil(max_y - min_y))
        out_w = max(out_w, 1)
        out_h = max(out_h, 1)

        HL = T.dot(Minv)
        HR = T.dot(M)

        warped_left = cv2.warpPerspective(img_left, HL, (out_w, out_h), flags=cv2.INTER_LINEAR)
        warped_right = cv2.warpPerspective(img_right, HR, (out_w, out_h), flags=cv2.INTER_LINEAR)

        # 混合影像
        if blender is not None:
            out = blender.linearBlending([warped_left, warped_right])
        else:
            out = self._simple_blend(warped_left, warped_right)
        
        out = self.removeBlackBorder(out)
        return out

    def _simple_stitch(self, img_left, img_right, H):
        """簡單的拼接方法（fallback）"""
        hl, wl = img_left.shape[:2]
        hr, wr = img_right.shape[:2]
        
        # 變換右圖
        warped_right = cv2.warpPerspective(img_right, H, (wl + wr, max(hl, hr)))
        
        # 建立輸出影像
        out = warped_right.copy()
        out[:hl, :wl] = img_left
        
        return self.removeBlackBorder(out)

    def _simple_blend(self, img1, img2):
        """簡單的影像混合"""
        # 找到重疊區域
        mask1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) > 0
        mask2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) > 0
        overlap = mask1 & mask2
        
        # 混合影像
        out = img1.copy()
        out[mask2 & ~mask1] = img2[mask2 & ~mask1]
        out[overlap] = (img1[overlap].astype(np.float32) + img2[overlap].astype(np.float32)) / 2
        
        return out.astype(np.uint8)


if __name__ == "__main__":
    # 左右相機 index
    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(1)

    # 設定解析度
    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # 固定 Homography (3x3)
    HomoMat = np.array([
        [ 1.22399508e-01, -2.80766372e-02,  8.78576870e+02],
        [-6.89163165e-01,  9.09170037e-01,  6.09774704e+00],
        [-7.83628153e-04, -3.24714445e-06,  1.00000000e+00]
    ])

    stitcher = ImageStitcher()

    print("開始影像拼接...")
    print("按 'q' 鍵退出")

    while True:
        ret_left, img_left = cap_left.read()
        ret_right, img_right = cap_right.read()

        if not ret_left or not ret_right:
            print("⚠️ 相機影像讀取失敗")
            break

        # 左右相機旋轉
        img_left = cv2.rotate(img_left, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_right = cv2.rotate(img_right, cv2.ROTATE_90_CLOCKWISE)

        stitched = stitcher.warp((img_left, img_right), HomoMat)
        if stitched is None:
            print("⚠️ 拼接失敗")
            continue

        # 縮放顯示（如果影像太大）
        h, w = stitched.shape[:2]
        if w > 1200:
            scale = 1200 / w
            new_w, new_h = int(w * scale), int(h * scale)
            stitched_display = cv2.resize(stitched, (new_w, new_h))
        else:
            stitched_display = stitched

        cv2.imshow("Panorama", stitched_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()