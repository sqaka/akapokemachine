#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import tensorflow as tf
import pdb

# 識別ラベルの数(今回はihy:0,izm:1,kzm:2で、計3)
NUM_CLASSES = 3
# 学習する時の画像のサイズ(px)
IMAGE_SIZE = 28
# 画像の次元数(28* 28*カラー(?))
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

# 学習に必要なデータのpathや学習の規模を設定
# パラメタの設定、デフォルト値やヘルプ画面の説明文を登録できるTensorFlow組み込み関数
flags = tf.app.flags
FLAGS = flags.FLAGS
# 学習用データ
flags.DEFINE_string('train', './train.txt', 'File name of train data')
# 検証用テストデータ
flags.DEFINE_string('test', './test.txt', 'File name of train data')
# データを置いてあるフォルダ
flags.DEFINE_string('train_dir', './data/train/', 'Directory to put the training data.')
# データ学習訓練の試行回数
flags.DEFINE_integer('max_steps', 200, 'Number of steps to run trainer.')
# 1回の学習で何枚の画像を使うか
flags.DEFINE_integer('batch_size', 120, 'Batch size Must divide evenly into the dataset sizes.')
# 学習率、小さすぎると学習が進まないし、大きすぎても誤差が収束しなかったり発散したりしてダメとか
flags.DEFINE_float('learning_rate', 1e-6, 'Initial learning rate.')

# AIの学習モデル部分(ニューラルネットワーク)を作成する
# images_placeholder: 画像のplaceholder, keep_prob: dropout率のplace_holderが引数になり
# 入力画像に対して、各ラベルの確率を出力して返す
def inference(images_placeholder, keep_prob):
  x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])
  print(x_image)

  with tf.name_scope('conv1') as scope:
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    print(h_conv1)

  with tf.name_scope('pool1') as scope:
    h_pool1 = max_pool_2x2(h_conv1)
    print(h_pool1)

  with tf.name_scope('conv2') as scope:
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    print(h_conv2)

  with tf.name_scope('pool2') as scope:
    h_pool2 = max_pool_2x2(h_conv2)
    print(h_pool2)

  with tf.name_scope('fc1') as scope:
    w = int(IMAGE_SIZE / 4)
    W_fc1 = weight_variable([w * w * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, w * w * 64])
    print(h_pool2_flat)
    h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    h_fc1_drop = tf.nn.dropout(tf.nn.relu(h_fc1), keep_prob)
    print(h_fc1_drop)

  with tf.name_scope('fc2') as scope:
    W_fc2 = weight_variable([1024, NUM_CLASSES])
    b_fc2 = bias_variable([NUM_CLASSES])
    h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    print(h_fc2)

  with tf.name_scope('softmax') as scope:
    y_conv = tf.nn.softmax(h_fc2)
    print(y_conv)

  return y_conv


def inference_deep(images_placeholder, keep_prob):
  x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])
  print(x_image)

  with tf.name_scope('conv1') as scope:
    W_conv1 = weight_variable([3, 3, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    print(h_conv1)

  with tf.name_scope('pool1') as scope:
    h_pool1 = max_pool_2x2(h_conv1)
    print(h_pool1)

  with tf.name_scope('conv2') as scope:
    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    print(h_conv2)

  with tf.name_scope('pool2') as scope:
    h_pool2 = max_pool_2x2(h_conv2)
    print(h_pool2)

  with tf.name_scope('conv3') as scope:
    W_conv3 = weight_variable([3, 3, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    print(h_conv3)

  with tf.name_scope('pool3') as scope:
    h_pool3 = max_pool_2x2(h_conv3)
    print(h_pool3)

  with tf.name_scope('fc1') as scope:
    w = IMAGE_SIZE / pow(2, 3)
    W_fc1 = weight_variable([w * w * 128, 1024])
    b_fc1 = bias_variable([1024])
    h_pool3_flat = tf.reshape(h_pool3, [-1, w * w * 128])
    print(h_pool3_flat)
    h_fc1 = tf.matmul(h_pool3_flat, W_fc1) + b_fc1
    h_fc1_drop = tf.nn.dropout(tf.nn.relu(h_fc1), keep_prob)
    print(h_fc1_drop)

  with tf.name_scope('fc2') as scope:
    W_fc2 = weight_variable([1024, NUM_CLASSES])
    b_fc2 = bias_variable([NUM_CLASSES])
    h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    print(h_fc2)

  with tf.name_scope('softmax') as scope:
    y_conv = tf.nn.softmax(h_fc2)
    print(y_conv)

  return y_conv


def weight_variable(shape):
  # shape = tf.cast(shape, tf.int32)
  initial = tf.random.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  # shape = tf.cast(shape, tf.int32)
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def tf_print(tensor, name):
  print(name, tensor)
  return tensor


def max_pool_2x2(x):
  return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def loss(logits, labels):
  # http://qiita.com/ikki8412/items/3846697668fc37e3b7e0
  # cross_entropy = -tf.reduce_sum(labels * tf.log(logits))
  cross_entropy = -tf.reduce_sum(labels * tf.math.log(tf.clip_by_value(logits, 1e-10, 1)))
  tf.compat.v1.summary.scalar("cross_entropy", cross_entropy)
  return cross_entropy


def training(loss, learning_rate):
  train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
  return train_step


def accuracy(logits, labels):
  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  tf.compat.v1.summary.scalar("accuracy", accuracy)
  return accuracy

if __name__ == '__main__':
  # ファイルを開く
  f = open(FLAGS.train, 'r')
  # データを入れる配列
  train_image = []
  train_label = []
  path = "./data/train/"
  for i, line in enumerate(f):
    print("index:%d"%(i))
    # 改行を除いてスペース区切りにする
    line = line.rstrip()
    l = line.split()
    data_name = path + l[0]
    # データを読み込んで28x28に縮小
    img = cv2.imread(data_name)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    # 一列にした後、0-1のfloat値にする
    train_image.append(img.flatten().astype(np.float32)/255.0)
    # ラベルを1-of-k方式で用意する
    tmp = np.zeros(NUM_CLASSES)
    tmp[int(l[1])] = 1
    train_label.append(tmp)

  # numpy形式に変換
  train_image = np.asarray(train_image)
  train_label = np.asarray(train_label)
  f.close()

  path_test = "./data/test/"
  f = open(FLAGS.test, 'r')
  test_image = []
  test_label = []
  for line in f:
    line = line.rstrip()
    l = line.split()
    data_name = path_test + l[0]
    img = cv2.imread(data_name)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    test_image.append(img.flatten().astype(np.float32)/255.0)
    tmp = np.zeros(NUM_CLASSES)
    tmp[int(l[1])] = 1
    test_label.append(tmp)
  test_image = np.asarray(test_image)
  test_label = np.asarray(test_label)
  f.close()

  #TensorBoardのグラフに出力するスコープを指定
  with tf.Graph().as_default():
    # 画像を入れるためのTensor(28*28*3(IMAGE_PIXELS)次元の画像が任意の枚数(None)分はいる)
    images_placeholder = tf.compat.v1.placeholder("float", shape=(None, IMAGE_PIXELS))
    # ラベルを入れるためのTensor(3(NUM_CLASSES)次元のラベルが任意の枚数(None)分入る)
    labels_placeholder = tf.compat.v1.placeholder("float", shape=(None, NUM_CLASSES))
    # dropout率を入れる仮のTensor
    keep_prob = tf.compat.v1.placeholder("float")

    # inference2 = (images_placeholder, keep_prob)
    # inference()を呼び出してモデルを作る
    logits = inference(images_placeholder, keep_prob)
    # loss()を呼び出して損失を計算
    loss_value = loss(logits, labels_placeholder)
    #pdb.set_trace()
    # training()を呼び出して訓練して学習モデルのパラメーターを調整する
    train_op = training(loss_value, FLAGS.learning_rate)
    # 精度の計算
    acc = accuracy(logits, labels_placeholder)


    # 保存の準備
    saver = tf.compat.v1.train.Saver()
    # Sessionの作成(TensorFlowの計算は絶対Sessionの中でやらなきゃだめ)
    sess = tf.compat.v1.Session()
    # 変数の初期化(Sessionを開始したらまず初期化)
    sess.run(tf.initialize_all_variables())
    # TensorBoard表示の設定(TensorBoardの宣言的な?)
    summary_op = tf.compat.v1.summary.merge_all()
    # train_dirでTensorBoardログを出力するpathを指定
    summary_writer = tf.compat.v1.summary.FileWriter(FLAGS.train_dir, sess.graph_def)

    # 実際にmax_stepの回数だけ訓練の実行していく
    for step in range(FLAGS.max_steps):
      for i in range(len(train_image)//FLAGS.batch_size):
        # batch_size分の画像に対して訓練の実行
        # batch = [batch_file for batch_file in batch_files]
        batch = FLAGS.batch_size*i
        # feed_dictでplaceholderに入れるデータを指定する
        sess.run(train_op, feed_dict={
          images_placeholder: train_image[batch:batch+FLAGS.batch_size],
          labels_placeholder: train_label[batch:batch+FLAGS.batch_size],
          keep_prob: 0.5})

      # 1step終わるたびに精度を計算する
      train_accuracy = sess.run(acc, feed_dict={
        images_placeholder: train_image,
        labels_placeholder: train_label,
        keep_prob: 1.0})
      print("step %d, training accuracy %g"%(step, train_accuracy))

      # 1step終わるたびにTensorBoardに表示する値を追加する
      summary_str = sess.run(summary_op, feed_dict={
        images_placeholder: train_image,
        labels_placeholder: train_label,
        keep_prob: 1.0})
      summary_writer.add_summary(summary_str, step)

  # 訓練が終了したらテストデータに対する精度を表示する
  print("test accuracy %g"%sess.run(acc, feed_dict={
    images_placeholder: test_image,
    labels_placeholder: test_label,
    keep_prob: 1.0}))

  # データを学習して最終的に出来上がったモデルを保存
  # "model.ckpt"は出力されるファイル名
  # save_path = saver.save(sess, "model.ckpt")
  saver.save(sess, "model.ckpt")
