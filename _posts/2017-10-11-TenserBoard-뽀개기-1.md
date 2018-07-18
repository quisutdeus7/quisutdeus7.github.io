---
layout: post, page
title: 'Tensorboard 뽀개기(1)'
categories: 'CE study'
---

## 1. TensorBoard란?
TensorFlow에 기록된 로그들을 토대로 그래프로 시각화 해주는 도구로서 TensorFlow의 결과 비교나 변화를 보기 쉽게 해준다.

## 2. TensorBoard 실행 준비
```python
_summary = tf.summary.scalar('cost',cost)
.
.
.
merge = tf.summary.merge_all()
write = tf.summary.FileWriter('./path to logs',sess.graph)
.
.
_summary = sess.run(merge, feed_dict={...})
write.add_summary(_summary, global_step)
```  
위 형식은 TensorBoard를 사용하는데 필요한 메소드들을 나열한 것이다. (내 나름대로 생각한 메소드들)

## 3. TensorBoard 실행시키기
``` commandline  
tensorboard --logdir= path to logdir   
```   
위의 명령어는  
  
_'루트 폴더 아래의 특정 경로에 있는 directory안의 로그파일을 보겠다'_  
  
라는 의미의 코드로 tensorflow 소스코드 안에 작성한 FileWriter 함수 parameter 들 중에 경로값과 일치해야 한다.
그리고 port를 따로 지정하지 않았기에
```commandline
localhost:6006
```  
으로 연결된다. 그래서 port를 설정하고 싶다면  
``` commandline  
tensorboard --logdir= path to logdir --port=NNNN 
```  
처럼 마지막에 특정 port번호를 입력해 주면 된다.  

※ 여기서 경로 설정을 잘못하면 graph가 제대로 출력되지 않을 수 있다. *(ex. 빈화면)*  
   
> 참고한 사이트 : http://tensorflowstepbystep.tistory.com/4

## 4. TensorBoard 중요 Method  
*※ [2. TensorBoard 실행시키기]()에서 언급한 Method들을 예제코드와 [Tensorflow API r1.3](https://www.tensorflow.org/api_docs/) 의 분석 삽질로 나름 해석해 보았다.*

- __tf.summary.scalar('value name', value)__  
    : 이 함수는 수집하고자 하는 data를 지정하는 함수로 input parameter는 수집하고자 하는 값의 이름과 그 값을 가지고 있는 value를 받는다.
    
- __tf.summary.merge_all()__
    : tensorboard에 표시해주기 위해 tensor들을 수집하는 함수이다.
    이는 각각의 summary한 node들을 run시켜줄 때 효율성을 높이기 위해 미리 하나의 data로 묶어주는 함수이다.
    
- __tf.summary.FileWriter('./path to logs',sess.graph)__
    : summary하면서 발생한 tensor값들과 그래프를 로컬에 저장하기 위한 함수이다.
    input parameter로 log파일이 저장될 경로와 저장할 graph가 있다. 그 외 다른 parameter가 있지만 현재 학습과정에선 저 2가지만 기억하려 한다.
        

- __Writer.add_summary()__  
    : 말 그대로 summary들을 합치는 코드이다. 
```python
_summary = sess.run(merge, feed_dict={...})
write.add_summary(_summary, global_step)
```
위 코드처럼 merge_all한 summary protocol buffer들을 합치는 과정으로
step 하면서 값들을 저장한다.