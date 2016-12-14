import tensorflow as tf
#tf tensorflow 선언

from tensorflow.examples.tutorials.mnist import input_data
#Yann LeCun의 웹사이트에서 제공된 데이터

# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)
#mnist 데이터를 로딩한다.

# Set up model
x = tf.placeholder(tf.float32, [None, 784])
#심볼릭 변수들을 사용하여 상호작용하는 작업들을 기술

W = tf.Variable(tf.zeros([784, 10]))
#가중치를 선언

b = tf.Variable(tf.zeros([10]))
#편향을 선언

y = tf.nn.softmax(tf.matmul(x, W) + b)
#첫번째로 우리 입력이 특정 클래스에 해당되는지에 대한 증거를 더하고 그 다음 증거를 확률로 변환합니다.

y_ = tf.placeholder(tf.float32, [None, 10])
#교차 엔트로피를 구현하기 위해 우리는 우선적으로 정답을 입력하기 위한 새 placeholder를 추가

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#교차 엔트로피 생성

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#학습 스텝

# Session
init = tf.initialize_all_variables()
#모든 변수 초기화

sess = tf.Session()
sess.run(init)
#세션에서 모델을 시작하고 변수들을 초기화하는 작업을 실행

# Learning
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#1000번 동안 학습을 시킴.

# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#예측이 실제와 맞았는지 확인

# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#예측값 출력.