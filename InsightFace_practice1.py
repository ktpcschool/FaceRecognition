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
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)

            if face.gender is not None and face.age is not None:
                cv2.putText(dimg, '%s,%d' % (face.sex, face.age),
                            (box[0] - 1, box[1] - 4), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (0, 255, 0), 1)
        return dimg


def main():
    app = FaceAnalysis1()
    app.prepare(ctx_id=0, det_size=(640, 640))

    image_file = "input.jpg"
    img = cv2.imread(image_file)
    faces = app.get(np.asarray(img))
    print("number of faces:" + str(len(faces)))

    rimg = app.draw_on(img, faces)
    now = datetime.now()
    now_str = now.strftime("%y%m%d%H%M%S")
    cv2.imwrite(f"output{now_str}.jpg", rimg)


if __name__ == "__main__":
    main()
