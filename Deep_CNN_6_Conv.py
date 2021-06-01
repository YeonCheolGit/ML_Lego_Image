import tensorflow as tf
import random
import load_data
import matplotlib.pyplot as plt

# label number to label name
def convert_label_name(label):
    label_names = ["2357 Brick corner 1x2x2", "3003 Brick 2x2", "3004 Brick 1x2", "3005 Brick 1x1", "3022 Plate 2x2", "3023 Plate 1x2",
                   "3024 Plate 1x1", "3040 Roof Tile 1x2x45deg", "3069 Flat Tile 1x2", "3673 Peg 2M", "3713 Bush for Cross Axle", "3794 Plate 1X2 with 1 Knob",
                   "6632 Technic Lever 3M", "11214 Bush 3M friction with Cross axle", "18651 Cross Axle 2M with Snap friction", "32123 half Bush"]
    return label_names[int(label)]

tf.set_random_seed(779)  # for reproducibility

# load data
try:
    train_file_path = './train_data.csv'
    test_file_path = './test_data.csv'
    train_datas = load_data.load_and_preprocess_data(train_file_path)
    test_datas = load_data.load_and_preprocess_data(test_file_path)
except:
    print('우선 preprocess.py를 실행하셔야 합니다.')
    exit(1)

print('train Input X Shape:' + str(train_datas.data[...,:-16].data.shape))
print('train Label Y Shape:' + str(train_datas.data[...,-16:].data.shape))

#length of data
len_training_data = len(train_datas.data)
len_test_data =len(test_datas.data)
nb_classes = 16

# parameters
training_epochs = 6000
batch_size = 2500
learning_rate = 0.001

keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, 2500])
X_img = tf.reshape(X, [-1, 50, 50, 1])
Y = tf.placeholder(tf.float32, [None, nb_classes])

# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01))
L5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME')
L5 = tf.nn.relu(L5)
L5 = tf.nn.max_pool(L5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L5 = tf.nn.dropout(L5, keep_prob=keep_prob)
L5_flat = tf.reshape(L5, [-1, 512 * 2 * 2])

W6 = tf.get_variable("W6", shape=[512 * 2 * 2, 2500], initializer=tf.contrib.layers.xavier_initializer()) ## 마지막 weight층 출력 개수 증가
b6 = tf.Variable(tf.random_normal([2500]))
L6 = tf.nn.relu(tf.matmul(L5_flat, W6) + b6)
L6 = tf.nn.dropout(L6, keep_prob=keep_prob)

W7 = tf.get_variable("W7", shape=[2500, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.matmul(L6, W7) + b7

# Cost function & Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(len_training_data / batch_size)

        for i in range(total_batch):
            # 원본 학습 데이터로부터 batch size만큼 추출해가며 학습
            batch = train_datas.next_batch(batch_size=batch_size)
            batch_x = batch[...,:-16]
            batch_y = batch[...,-16:]

            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.7})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")

    # Test the model using test sets
    avg_acc=0
    batch_size = 512
    total_batch = int(len_test_data / batch_size)

    for i in range(total_batch):
        # 원본 테스트 데이터로부터 batch size만큼 추출해가며 정확도 계산
        batch = test_datas.next_batch(batch_size=batch_size)
        batch_x = batch[...,:-16]
        batch_y = batch[...,-16:]
        acc = sess.run(accuracy, feed_dict={X:batch_x, Y: batch_y, keep_prob: 1})
        avg_acc += acc / total_batch

    print("Accuracy: ", avg_acc)


    # Get one and predict
    print("\n<무작위로 1개 추출 후 예측>")
    r = random.randint(0, len_test_data - 1)
    one_data = test_datas.data[r:r+1]

    # get label value
    label = sess.run(tf.argmax(one_data[...,-16:], 1))  # [0,0,1...0] -> 3 변환
    label_name = convert_label_name(label)  # 0 -> 2357 Brick corner 1x2x2 변환
    print("Label: ", label, label_name)

    # get predict value
    pred = sess.run(tf.argmax(hypothesis, 1), feed_dict={X: one_data[...,:-16], keep_prob: 1})
    pred_name = convert_label_name(pred)
    print("Prediction: ", pred, pred_name)

    # numpy array로 부터 이미지를 출력하는 부분
    plt.imshow(
        one_data[...,:-16].reshape(50,50),
        cmap='Greys',
        interpolation='nearest')
    plt.show()


