import numpy as np
import cv2
from matplotlib import pyplot as plt


class Blender:
    def linearBlending(self, imgs):
        '''
        linear Blending(also known as Feathering)
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]

        # Non-black masks
        left_mask = np.any(img_left != 0, axis=2)
        right_mask = np.any(img_right != 0, axis=2)
        overlap = left_mask & right_mask

        # Plot the overlap mask (kept for parity with original behavior)
        plt.figure(21)
        plt.title("overlap_mask")
        plt.imshow(overlap.astype(int), cmap="gray")

        # If no overlap, simply composite
        if not np.any(overlap):
            out = img_right.copy()
            out[left_mask] = img_left[left_mask]
            return out

        # Compute first and last overlap columns per row
        row_has_overlap = np.any(overlap, axis=1)
        first = np.zeros(hr, dtype=int)
        last = np.zeros(hr, dtype=int)
        first[row_has_overlap] = np.argmax(overlap[row_has_overlap], axis=1)
        last[row_has_overlap] = wr - 1 - np.argmax(overlap[row_has_overlap][:, ::-1], axis=1)

        length = np.maximum(1, (last - first))
        j = np.arange(wr)[None, :]
        first_col = first[:, None]
        length_col = length[:, None]

        alpha = np.zeros((hr, wr), dtype=np.float32)
        inside = (j >= first_col) & (j <= (last[:, None])) & row_has_overlap[:, None]
        ramp = 1.0 - (j - first_col) / length_col
        alpha[inside] = ramp[inside]

        out = img_right.astype(np.float32).copy()
        only_left = left_mask & ~right_mask
        out[only_left] = img_left[only_left]

        a = alpha[..., None]
        out[overlap] = a[overlap] * img_left[overlap] + (1.0 - a[overlap]) * img_right[overlap]

        return np.clip(out, 0, 255).astype(np.uint8)

    def linearBlendingWithConstantWidth(self, imgs):
        '''
        linear Blending with Constat Width, avoiding ghost region
        # you need to determine the size of constant with
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        constant_width = 3

        # Non-black masks
        left_mask = np.any(img_left != 0, axis=2)
        right_mask = np.any(img_right != 0, axis=2)
        overlap = left_mask & right_mask

        if not np.any(overlap):
            out = img_right.copy()
            out[left_mask] = img_left[left_mask]
            return out

        row_has_overlap = np.any(overlap, axis=1)
        first = np.zeros(hr, dtype=int)
        last = np.zeros(hr, dtype=int)
        first[row_has_overlap] = np.argmax(overlap[row_has_overlap], axis=1)
        last[row_has_overlap] = wr - 1 - np.argmax(overlap[row_has_overlap][:, ::-1], axis=1)

        mid = (first + last) // 2
        j = np.arange(wr)[None, :]
        mid_col = mid[:, None]
        first_col = first[:, None]
        last_col = last[:, None]

        alpha = np.zeros((hr, wr), dtype=np.float32)
        # Regions fully left/right of the blending band
        alpha = np.where(j <= (mid_col - constant_width), 1.0, alpha)
        alpha = np.where(j >= (mid_col + constant_width), 0.0, alpha)

        # Linear ramp inside the band
        band = (j > (mid_col - constant_width)) & (j < (mid_col + constant_width)) & row_has_overlap[:, None]
        alpha_band = 1.0 - (j - (mid_col - constant_width)) / (2.0 * constant_width)
        alpha = np.where(band, alpha_band, alpha)

        # Zero out columns outside the overlap bounds
        valid = (j >= first_col) & (j <= last_col) & row_has_overlap[:, None]
        alpha = np.where(valid, alpha, 0.0)

        out = img_right.astype(np.float32).copy()
        only_left = left_mask & ~right_mask
        out[only_left] = img_left[only_left]

        a = alpha[..., None]
        out[overlap] = a[overlap] * img_left[overlap] + (1.0 - a[overlap]) * img_right[overlap]

        return np.clip(out, 0, 255).astype(np.uint8)