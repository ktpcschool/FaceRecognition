"""
Copyright (c) 2018 Jiankang Deng and Jia Guo
License: MIT
https://github.com/deepinsight/insightface

Copyright (c) 2020 KazuhitoTakahashi
License: MIT
https://github.com/Kazuhito00/cvoverlayimg

Copyright(c) 2022 Tatsuro Watanabe
License: MIT
https://github.com/ktpcschool/FaceRecognition
"""
import cv2
from datetime import datetime
import numpy as np

from insightface.app import FaceAnalysis
from PIL import Image


class CvOverlayImage(object):
    """
    [summary]
      OpenCV形式の画像に指定画像を重ねる
    """

    def __init__(self):
        pass

    @classmethod
    def overlay(
            cls,
            cv_background_image,
            cv_overlay_image,
            point,
    ):
        """
        [summary]
          OpenCV形式の画像に指定画像を重ねる
        Parameters
        ----------
        cv_background_image : [OpenCV Image]
        cv_overlay_image : [OpenCV Image]
        point : [(x, y)]
        Returns : [OpenCV Image]
        """
        overlay_height, overlay_width = cv_overlay_image.shape[:2]

        # OpenCV形式の画像をPIL形式に変換(α値含む)
        # 背景画像
        cv_rgb_bg_image = cv2.cvtColor(cv_background_image, cv2.COLOR_BGR2RGB)
        pil_rgb_bg_image = Image.fromarray(cv_rgb_bg_image)
        pil_rgba_bg_image = pil_rgb_bg_image.convert('RGBA')
        # オーバーレイ画像
        cv_rgb_ol_image = cv2.cvtColor(cv_overlay_image, cv2.COLOR_BGRA2RGBA)
        pil_rgb_ol_image = Image.fromarray(cv_rgb_ol_image)
        pil_rgba_ol_image = pil_rgb_ol_image.convert('RGBA')

        # composite()は同サイズ画像同士が必須のため、合成用画像を用意
        pil_rgba_bg_temp = Image.new('RGBA', pil_rgba_bg_image.size,
                                     (255, 255, 255, 0))
        # 座標を指定し重ね合わせる
        pil_rgba_bg_temp.paste(pil_rgba_ol_image, point, pil_rgba_ol_image)
        result_image = \
            Image.alpha_composite(pil_rgba_bg_image, pil_rgba_bg_temp)

        # OpenCV形式画像へ変換
        cv_bgr_result_image = cv2.cvtColor(np.asarray(result_image),
                                           cv2.COLOR_RGBA2BGRA)

        return cv_bgr_result_image


# FaceAnalysisを継承（draw_on関数を上書き）
class FaceAnalysis1(FaceAnalysis):
    def draw_on(self, img, faces, mosaic_img):
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int)

            # 認識した部分の画像にモザイクをかける
            dimg = mosaic(dimg, (box[0], box[1], box[2], box[3]), mosaic_img)

        return dimg


def mosaic(img, rect, cv_overlay_image):
    # モザイクをかける領域を取得
    (x1, y1, x2, y2) = rect
    w = x2 - x1
    h = y2 - y1

    cv_background_image = img.copy()

    # モザイク処理のため画像サイズを変換する
    cv_overlay_image = cv2.resize(cv_overlay_image, (w, h))

    point = (x1, y1)
    img2 = CvOverlayImage.overlay(cv_background_image,
                                  cv_overlay_image,
                                  point)
    return img2


def main():
    app = FaceAnalysis1()
    app.prepare(ctx_id=0, det_size=(640, 640))

    image_file = "input.jpg"
    img = cv2.imread(image_file)
    faces = app.get(np.asarray(img))
    mosaic_image_file = "smiley.png"
    mosaic_img = cv2.imread(mosaic_image_file,
                            cv2.IMREAD_UNCHANGED)  # IMREAD_UNCHANGEDを指定しα込みで読み込む
    rimg = app.draw_on(img, faces, mosaic_img)
    now = datetime.now()
    now_str = now.strftime("%y%m%d%H%M%S")
    cv2.imwrite(f"output{now_str}.jpg", rimg)
    print("完了しました。")


if __name__ == "__main__":
    main()
