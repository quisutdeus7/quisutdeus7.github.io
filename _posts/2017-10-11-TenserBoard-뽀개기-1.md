<a> *Tensorboard 뽀개기(1)* </a>
==================================

## 1. TensorBoard란?
TensorFlow에 기록된 로그들을 토대로 그래프로 시각화 해주는 도구로서  
TensorFlow의 결과 비교나 변화를 보기 쉽게 해준다.

## 2. TensorBoard 실행 준비


## 3. TensorBoard 실행시키기
```anaconda commandline
tensorboard --logdir= path to logdir   
```   
위의 명령어는  
'루트 폴더 아래의 특정 경로에 있는 directory안의 로그파일을 보겠다'  
라는 의미의 코드로 tensorflow 소스코드 안에 작성한 FileWriter 함수 parameter 들 중에 경로값과 일치해야 한다.    
※ 여기서 경로 설정을 잘못하면 graph가 제대로 출력되지 않을 수 있다.(ex. 빈화면)

> 만약에 경로를 잘못 입력하면 다음과 같은 결과가 나온다.  
 ![터미널에서의 결과](https://github.com/quisutdeus7/quisutdeus7.github.io/_data/img/tensorboard_error_1.PNG)
