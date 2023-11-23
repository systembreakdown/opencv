import os
import cv2
import numpy as np
import glob
import pickle
# 自定義的intrinsic模組，請根據實際情況更改
from intrinsic_module import undistort, get_chessboard_mapping, find_intrinsic_params, plot_3d_eval

if __name__ == '__main__':
    WIDTH, HEIGHT = 710, 710  # 2840, 2840
    PIXEL_SIZE = 0.00274 * 4  # 毫米 (µm/1000)
    CHECKERBOARD = (7, 10)
    BLOCK_SIZE = 34  # 毫米
    pkl_dir = 'pkls'

    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)

    img_dir = 'chessboard'
    img_fps = glob.glob(os.path.join(img_dir, '*.png'))
    pkl_fp = os.path.join(pkl_dir,
                          os.path.basename(os.path.dirname(img_dir)) + "_" + os.path.basename(img_dir) + ".pkl")

    with open(pkl_fp, 'rb') as f:
        ret, mtx_ori, dist_src, rvecs, tvecs, mtx, roi = pickle.load(f)

    imgs = [undistort(cv2.imread(img_fp), mtx_ori, dist_src, mtx) for img_fp in img_fps]
    objpoints, imgpoints = get_chessboard_mapping(img_fps, CHECKERBOARD, BLOCK_SIZE, imgs=imgs, imshow=False)
    ret, mtx, dist, rvecs, tvecs, mtx_n, roi = find_intrinsic_params(objpoints, imgpoints, (HEIGHT, WIDTH))

    print("未校正的失真參數", dist_src)
    print('校正後的失真參數', dist)

    # 繪製3D圖
    plot_3d_eval(rvecs, tvecs, mtx, objpoints, imgpoints, WIDTH, HEIGHT, ax=None, scale_img=50,
                 title='所有校正圖像的三維空間重構')