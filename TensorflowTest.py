import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
'''
data = pd.read_csv('/Users/luoyu/Downloads/MNIST-DATA/train.csv')
data = data.values
train_data, test_data = train_test_split(data,test_size=0.2)

train_x = train_data[:,1:785].T
train_y = train_data[:,0:1].T
dev_x = test_data[0:4200,1:785].T
dev_y = test_data[0:4200,0:1].T
test_x = test_data[4200:8400,1:785].T
test_y = test_data[4200:8400,0:1].T
#print(train_x.shape,train_y.shape,dev_x.shape,dev_y.shape,test_x.shape,test_y.shape)

tmp = np.zeros([10,train_y.shape[1]])
for i in range(tmp.shape[1]):
    tmp[train_y[0,i],i]=1.0
train_y = tmp

tmp = np.zeros([10,dev_y.shape[1]])
for i in range(tmp.shape[1]):
    tmp[dev_y[0,i],i]=1.0
dev_y = tmp

np.savetxt('/Users/luoyu/Downloads/MNIST-DATA/my_train_x.csv',train_x.T[0:100], delimiter=",")
np.savetxt('/Users/luoyu/Downloads/MNIST-DATA/my_train_y.csv',train_y.T[0:100], delimiter=",")
'''
# Dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_x = mnist.train.images
train_y = mnist.train.labels
train_x = train_x.T
train_y = train_y.T
dev_x = mnist.validation.images
dev_y = mnist.validation.labels
dev_x = dev_x.T
dev_y = dev_y.T
#np.savetxt('/Users/luoyu/Downloads/MNIST-DATA/rrrr_train_x.csv',train_x.T[0:100], delimiter=",")
#np.savetxt('/Users/luoyu/Downloads/MNIST-DATA/rrrr_train_y.csv',train_y.T[0:100], delimiter=",")

# configuration
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER_NODE1 = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000

# construction
x = tf.placeholder(tf.float32,[INPUT_NODE,None])
y = tf.placeholder(tf.float32,[OUTPUT_NODE,None])
w1 = tf.Variable(tf.truncated_normal([LAYER_NODE1,INPUT_NODE],stddev=0.1),dtype=tf.float32,trainable=True)
w2 = tf.Variable(tf.truncated_normal([OUTPUT_NODE,LAYER_NODE1],stddev=0.1),dtype=tf.float32,trainable=True)
b1 = tf.Variable(tf.zeros([LAYER_NODE1,1]),dtype=tf.float32,trainable=True)
b2 = tf.Variable(tf.zeros([OUTPUT_NODE,1]),dtype=tf.float32,trainable=True)

# forward propagation
def forward(x,w1,w2,b1,b2):
    a = tf.nn.relu(tf.matmul(w1,x)+b1)
    h = tf.nn.softmax(tf.matmul(w2,a)+b2)
    return h

# backward propagation
h = forward(x,w1,w2,b1,b2)
global_step = tf.Variable(0)
rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,TRAINING_STEPS/BATCH_SIZE,LEARNING_RATE_DECAY)
regulization = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(w1)+tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(w2)
lost = -y*tf.log(tf.clip_by_value(h,1e-10,1))
cost = tf.reduce_mean(lost) + regulization
optimizer = tf.train.AdamOptimizer(learning_rate=rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
#train = tf.train.GradientDescentOptimizer(rate).minimize(loss=cost, global_step=global_step)
train = optimizer.minimize(loss=cost,global_step=global_step)
correct_prediction = tf.equal(tf.argmax(h,axis=0), tf.argmax(y,axis=0))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(TRAINING_STEPS):
        start = (i*BATCH_SIZE)%train_x.shape[1]
        end = min(start+BATCH_SIZE,train_x.shape[1])
        train_feed = {x: train_x[:,start:end], y: train_y[:,start:end]}
        dev_feed = {x:dev_x,y:dev_y}
        if i % 1000 == 0:
            acc,cost_function = sess.run([accuracy,cost], feed_dict=train_feed)
            print("After %d training steps, training accuracy is %g" %(i, acc))
            print("The cost function is %g" %cost_function)

        sess.run(train, feed_dict=train_feed)

    dev_acc = sess.run(accuracy,feed_dict=dev_feed)
    print("After training, the dev_set accuracy is %g" %dev_acc)