"""
Copyright (c) 2018 Jiankang Deng and Jia Guo
License: MIT
https://github.com/deepinsight/insightface

Copyright(c) 2022 Tatsuro Watanabe
License: MIT
https://github.com/ktpcschool/FaceRecognition
"""
import cv2
from datetime import datetime
import numpy as np

from insightface.app import FaceAnalysis


# FaceAnalysisを継承（draw_on関数を上書き）
class FaceAnalysis1(FaceAnalysis):
    def draw_on(self, img, faces):
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int)

            # 認識した部分の画像にぼかしをかける
            dimg = blur(dimg, (box[0], box[1], box[2], box[3]))

        return dimg


def blur(img, rect):
    # ぼかしをかける領域を取得
    (x1, y1, x2, y2) = rect
    i_rect = img[y1:y2, x1:x2]

    # ぼかし処理
    i_mos = cv2.blur(i_rect, ksize=(15, 15))

    # 画像にぼかし画像を重ねる
    img2 = img.copy()
    img2[y1:y2, x1:x2] = i_mos
    return img2


def main():
    app = FaceAnalysis1()
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
