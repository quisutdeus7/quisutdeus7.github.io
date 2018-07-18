---
layout: post, list
title: 'Tensorboard 뽀개기(2)'
categories: 'CE study'
---

> *※ 지난 [Tensorboard 뽀개기(1)](https://quisutdeus7.github.io/jekyll/update/2017/10/11/TenserBoard-%EB%BD%80%EA%B0%9C%EA%B8%B0-1/)에 이어서 실제 Tensorboad를 활용한 [예제](https://github.com/golbin/TensorFlow-Tutorials/tree/master/05%20-%20TensorBoard%2C%20Saver)를 구현해가면서 기능을 알아봄.*

## 5. 간단한 CNN으로 Tensorboard 적용하기

```text
0, 0, 1, 0, 0
1, 0, 0, 1, 0
1, 1, 0, 0, 1
0, 0, 1, 0, 0
0, 0, 1, 0, 0
0, 1, 0, 0, 1
```
먼저 데이터 파일은 위와 같은 형식의 csv로 local directory에 갖고 있어야 한다.
  
```python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# 적용시킬 data를 가져온다.
data = np.loadtxt('./data.csv', delimiter=',',
                  unpack=True, dtype='float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])
``` 
 1. [*coding: utf-8*]()  
    : 코드 내에 있는 한글 주석때문에 run할 때 SyntexError를 발생하는 경우가 있는데 이는 python의 기본 인코딩이 ascii이기에 euc-kr, utf-8 형식인 한글같은 경우 해석이 불가능하기 때문이다.
     그래서 한글 주석을 달아주기 위해 파일 상단에 [*coding: utf-8*]()를 넣어줘야지 다른 곳에서 삽질을 안한다. 
       
    ~~물론 영어 주석에 익숙한 coder는 상관없는 문제겠다.~~
    
 2. [*np.loadtxt('./data.csv', delimiter=',',unpack=True, dtype='float32')*]()
	: Numpy 패키지에 있는 loadtxt 함수로 로컬에 저장되있는 숫자형식(이 부분은 확인이 더 필요하다)데이터가 저장된 파일을 load한다. 불러올 때 해당 정보의 타입이나 자료 구분자 등에 대한 정보는 input parameter로 지정해준다.

 3.  [*np.transpose(data[a:b])*]()
	: 배열의 행과 열을 바꾸는 함수로 python의 슬라이싱을 이용해 원형 배열에서도 일부만 transpose를 할수 있게 해준다. 여기서는 원형 데이터를 같고있는 data배열에서 index가 [0:2]인 부분은 xdata, 나머지는 ydata로 사용했다.  
	
     
```python
####
# 신경망 모델 구성
####
global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# with tf.name_scope 으로 묶은 블럭은 텐서보드에서 한 레이어안에 표현해줌
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.), name='W1')
    L1 = tf.nn.relu(tf.matmul(X, W1))

    # 추가코드
    tf.summary.histogram("X", X)
    tf.summary.histogram("Weights", W1)

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.), name='W2')
    L2 = tf.nn.relu(tf.matmul(L1, W2))

    # 추가코드
    tf.summary.histogram("Weights", W2)

with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.), name='W3')
    model = tf.matmul(L2, W3)

    # 추가코드
    tf.summary.histogram("Weights", W3)
    tf.summary.histogram("Model", model)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost, global_step=global_step)

    # tf.summary.scalar 를 이용해 수집하고 싶은 값들을 지정할 수 있습니다.
    tf.summary.scalar('cost', cost)
```
 4. [*with tf.name_scope('put a layer name'):*]()  
    : 여기서는 with와 name_scope함수를 통해 Tensorboard에서 graph가 얼마나 간소화 되는지 알 수 있다.
    
 5. [*tf.summary.histogram("Weights", W1)*]()  
    : TensorBoard의 Histogram메뉴에서 그래프가 그려지도록 하는 함수이다. 
 
 6. [*tf.summary.scalar('cost', cost)*]()  
    : _summary.scalar는 앞장에서 설명했기에 생략한다._


#### ※ 쓸데없을 수 있지만 고민한 흔적들
- 현재 review중인 code의 cost가 왜 0.550에 머무는가
- with와 name_scope를 사용여부에 따른 그래프의 형태 변화
- 학습이 안되는 case에 대한 비교할 방도
- histogram 함수도 with문 안에 있는데 왜 Tensorboard의 graph에 표현이 안되는가
- Histaograms에 그려져 있는 그래프들의 의미. 그리고 왜 epoch 과정중 일부가 표현이 안되어 있는가