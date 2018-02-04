import tensorflow as tf
from numpy import *
from tensorflow.examples.tutorials.mnist import input_data

# configuration of ANN

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER_NODE1 = 500

BATCH_SIZE = 128

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000

# forward propagation
def fp(x, weights1, biases1, weights2, biases2):
    layer1 = tf.nn.relu(tf.matmul(x, weights1)+biases1)
    return tf.matmul(layer1, weights2) + biases2

def train(training_size, tx, ty, cx, cy):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-output')

    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER_NODE1], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER_NODE1]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER_NODE1, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    h = fp(x, weights1, biases1, weights2, biases2)
    step = tf.Variable(0)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h, labels=tf.cast(tf.argmax(y, 1), tf.int32))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean+regularization
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, step, training_size/BATCH_SIZE, LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=step)
    correct_prediction = tf.equal(tf.argmax(h,1), tf.cast(tf.argmax(y, 1), tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed = {x: cx, y: cy}
        for i in range(TRAINING_STEPS):
            s = (i * BATCH_SIZE) % training_size
            e = min(s+BATCH_SIZE, training_size)

            if i % 1000 == 0:
                acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training steps, validation accuracy is %g" %(i, acc))
            sess.run(train_step, feed_dict={x: tx[s:e], y: ty[s:e]})

list1 = []
list2 = []
'''
with open('/Users/luoyu/Downloads/train.csv', 'rt') as myFile:
    lines = csv.reader(myFile)
    for line in lines:
        if line[0] != 'label':
            list1.append(line[0])
            list2.append(line[1:len(line)])

cy = array(list1[0:8400]).astype('float32')
cx = array(list2[0:8400]).astype('float32')
ty = array(list1[8400:42000]).astype('float32')
tx = array(list2[8400:42000]).astype('float32')
'''
xs, ys = mnist.train.next_batch(BATCH_SIZE)

training_size = mnist.train.num_examples
print(training_size)

train(training_size, xs, ys, mnist.validation.images, mnist.validation.labels)
