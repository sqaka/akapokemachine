#　今回必要なや〜つ
import pyautogui
import time

# このフォルダに素材をブチ込む、をこちらで指定してやる
output_path = "./scammer_pic/"

# 1000枚写真を撮る
for i in range(1000):
    #　スクショ撮るや〜つ
    s = pyautogui.screenshot()
    # 上で定義したoutput_pathフォルダに4桁の連番数字がついた画像が入っていく
    s.save(output_path + 'scammer_{0:04d}.png'.format(i))
    #　0.5秒ごとに撮影される　と勝手に理解している
    time.sleep(0.5)
