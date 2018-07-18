---
layout: post
title: 'Tensorboard 뽀개기(3)'
categories: 'CE study'
---

> *※ 지난 [Tensorboard 뽀개기(2)](https://quisutdeus7.github.io/tensorboard/study/2017/10/12/Tensorboard-%EB%BD%80%EA%B0%9C%EA%B8%B0-2/)에 이어서 실제 Tensorboad를 활용한 [예제](https://github.com/golbin/TensorFlow-Tutorials/tree/master/05%20-%20TensorBoard%2C%20Saver)를 구현해가면서 기능을 알아봄.*

## 5. 간단한 CNN으로 Tensorboard 적용하기(이어서)

     
```python
#########
# 신경망 모델 학습
######
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')

if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)

# 최적화 진행
for step in range(2000):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    print('Step: %d, ' % sess.run(global_step),
          'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    # 적절한 시점에 저장할 값들을 수집하고 저장합니다.
    summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
    writer.add_summary(summary, global_step=sess.run(global_step))

saver.save(sess, './model/dnn.ckpt', global_step=global_step)

```
 7. [*saver = tf.train.Saver(tf.global_variables())*]()  
    : tensor값들을 저장하도록 하는 함수로 10번 함수인 save가 실행되도록 한다.
    
 8. [*ckpt = tf.train.get_checkpoint_state('./model')*]()  
    : 체크포인트를 할 폴더의 위치를 저장할 함수로 10번 함수인 save가 실행되도록 한다.
    
 9. [*merged = tf.summary.merge_all()*]()  
    [*writer = tf.summary.FileWriter('./logs', sess.graph)*]()  
    [*writer.add_summary(summary, global_step=sess.run(global_step))*]()  
    : _위 함수들은 1장에서 설명했기에 생략한다._
   
 10. [*saver.save(sess, './model/dnn.ckpt', global_step=global_step)*]()  
    : global_step인수에 값을 전달하여 체크포인트할 파일 이름의 수를 결정하는 함수이다. 

## 6. Tensorboard에 표시된 결과물
### 1) Main Graph  
<img src="https://raw.githubusercontent.com/quisutdeus7/quisutdeus7.github.io/master/_data/img/tensorboard_basic2_graph.PNG" width="70%" align = "" >

 위 그래프는 node들을 따로 정리하지 않아서 clear하지 않는 형태이다. Tensorboard 화면에서 *remove from main graph* 기능을 실행하면
 깔끔하게 정리가 가능하다.
  
### 2) Scalars
<img src="https://raw.githubusercontent.com/quisutdeus7/quisutdeus7.github.io/master/_data/img/tensorboard_basic2_scalar.PNG" width="70%">

### 3) Distributions
<img src="https://raw.githubusercontent.com/quisutdeus7/quisutdeus7.github.io/master/_data/img/tensorboard_basic2_distribute1.PNG" width="70%">

### 4) Histograms
<img src="https://raw.githubusercontent.com/quisutdeus7/quisutdeus7.github.io/master/_data/img/tensorboard_basic2_histogram1.PNG" width="70%">
<img src="https://raw.githubusercontent.com/quisutdeus7/quisutdeus7.github.io/master/_data/img/tensorboard_basic2_histogram3.PNG" width="70%">

## 6. 쓸데없을 수 있지만 고민한 흔적들 
- 현재 review중인 code의 cost가 왜 0.550에 머무는가
- with와 name_scope를 사용여부에 따른 그래프의 형태 변화  
    : with와 name_scope를 사용하지 않을 경우 그래프가 매우 퍼져있는(?) 형태가 되어 한눈에 파악하기가 힘들다
- 학습이 안되는 case에 대한 비교
    : Learning rate 변화만 먼저 시험.
- Histogram 함수도 with문 안에 있는데 왜 Tensorboard의 graph에 표현이 안되는가
- Histograms에 그려져 있는 그래프들의 의미. 그리고 왜 epoch 과정중 일부가 표현이 안되어 있는가

