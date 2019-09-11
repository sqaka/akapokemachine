# -*- coding:utf-8 -*-

import cv2
import glob

# 画像データのパス
input_data_path = './data/new_capture/scammer_tube/'
# 切出し後画像の保存先
save_path = './data/cutgazo/new_capture/scammer_tube/'
# OpenCVの顔分類器。直下に置きました
cascade_path = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)

files =glob.glob('./data/new_capture/scammer_tube/*')
face_detect_count = 0
rabel = "scammer"

# 検出した顔画像を切出して保存
for fname in files:
  img = cv2.imread(fname, cv2.IMREAD_COLOR)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  face = faceCascade.detectMultiScale(gray, 1.1, 3)
  
  # 1つでも顔を検出したら保存するよ
  if len(face) > 0:
    for rect in face:
      x = rect[0]
      y = rect[1]
      w = rect[2]
      h = rect[3]
      
      # save_pathに名前をつけて画像を保存する。ここかなり力技なので見た目きたないです
      cv2.imwrite(save_path + 'cut' + str(rabel) + str(face_detect_count) + '.png', img[y:y+h, x:x+w])
      face_detect_count = face_detect_count + 1

# 顔がなければコンソールにNoFaceと返す
  else:
    print(fname + ':NoFace')
