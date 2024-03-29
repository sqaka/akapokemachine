# -*- coding: utf-8 -*-

import tensorflow as tf
import multiprocessing as mp

from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from werkzeug import secure_filename
import os
import eval

# app(文字列はなんでもいい)名義でインスタンス化
app = Flask(__name__)
app.config['DEBUG'] = True
# 投稿画像の保存先
UPLOAD_FOLDER = './static/images/default/'

# ルートアクセス時の挙動を設定　なぜか'/'から"/"に変えるとエラーひとつ消える
@app.route("/")
def index():
    return render_template('index.html')

# 画像投稿時のアクション　ここのpostも文字列なんでもいいみたい
@app.route('/post', methods=['GET','POST'])
def post():
  if request.method == 'POST':
    if not request.files['file'].filename == u'':
        # アップロードされたファイルを保存
        f = request.files['file']
        img_path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
        f.save(img_path)
        # eval.pyへアップロードされた画像を渡す
        result = eval.evaluation(img_path, './model.ckpt')
    else:
        result = []
    return render_template('index.html', result=result)
  else:
    # エラーの際は「エラーです」等の気の利いたお知らせはせず無慈悲にトップに戻す
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
