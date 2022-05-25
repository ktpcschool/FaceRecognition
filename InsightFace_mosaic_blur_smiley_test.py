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
from PIL import Image

from insightface.app import FaceAnalysis


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
        cv_bgr_result_image = cv2.cvtColor(
            np.asarray(result_image), cv2.COLOR_RGBA2BGRA)

        return cv_bgr_result_image


# FaceAnalysisを継承（draw_on関数を上書き）
class FaceAnalysis1(FaceAnalysis):
    def __init__(self,
                 mosaic_method,
                 root='~/.insightface',
                 size=None,
                 cv_overlay_image=None):
        super().__init__(root=root)
        self.mosaic_method = mosaic_method
        self.size = size
        self.cv_overlay_image = cv_overlay_image

    def draw_on(self, img, faces):
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int)

            # 認識した部分の画像にモザイクをかける
            dimg = self.mosaic_method(dimg,
                                      (box[0], box[1], box[2], box[3]),
                                      self.size,
                                      self.cv_overlay_image)

        return dimg


def mosaic(img, rect, size, cv_overlay_image):
    # モザイクをかける領域を取得
    (x1, y1, x2, y2) = rect
    w = x2 - x1
    h = y2 - y1
    i_rect = img[y1:y2, x1:x2]

    # モザイク処理のため一度縮小して拡大する
    i_small = cv2.resize(i_rect, size)
    i_mos = cv2.resize(i_small, (w, h), interpolation=cv2.INTER_AREA)

    # 画像にモザイク画像を重ねる
    img2 = img.copy()
    img2[y1:y2, x1:x2] = i_mos
    return img2


def blur(img, rect, size, cv_overlay_image):
    # ぼかしをかける領域を取得
    (x1, y1, x2, y2) = rect
    i_rect = img[y1:y2, x1:x2]

    # ぼかし処理
    i_mos = cv2.blur(i_rect, ksize=size)

    # 画像にぼかし画像を重ねる
    img2 = img.copy()
    img2[y1:y2, x1:x2] = i_mos
    return img2


def mosaic_by_image(img, rect, size, cv_overlay_image):
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
    root = '~/.insightface'

    # モザイク
    # mosaic_method = mosaic
    # size = (10, 10)
    # cv_overlay_image = None

    # ぼかし
    # mosaic_method = blur
    # size = (15, 15)
    # cv_overlay_image = None

    # smiley
    mosaic_method = mosaic_by_image
    size = None
    mosaic_image_file = "smiley.png"
    mosaic_img = cv2.imread(mosaic_image_file,
                            cv2.IMREAD_UNCHANGED)  # IMREAD_UNCHANGEDを指定しα込みで読み込む
    cv_overlay_image = mosaic_img

    app = FaceAnalysis1(mosaic_method=mosaic_method,
                        root=root,
                        size=size,
                        cv_overlay_image=cv_overlay_image)
    app.prepare(ctx_id=0, det_size=(640, 640))

    image_file = "input.jpg"
    img = cv2.imread(image_file)
    faces = app.get(np.asarray(img))
    rimg = app.draw_on(img, faces)
    now = datetime.now()
    now_str = now.strftime("%y%m%d%H%M%S")
    cv2.imwrite(f"output{now_str}.jpg", rimg)
    print("完了しました。")


if __name__ == "__main__":
    main()
