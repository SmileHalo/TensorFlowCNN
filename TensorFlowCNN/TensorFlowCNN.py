import tensorflow as tf
from imageHandler import ImageHandler as ih
import random


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x,name):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME',name=name)

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name=name)


imageSize = 784 
x = tf.placeholder("float", [None, 784],name='x') #定义占位符 X为输入结构是 宽度400 的输入流
W = tf.Variable(tf.zeros([imageSize,10]),name='weight')  #定义运算变量 W为400*10的矩阵
b = tf.Variable(tf.zeros([10]),name='bias')
y = tf.nn.softmax(tf.matmul(x,W) + b,name='y') #定义激活函数 y=x*w+b
y_ = tf.placeholder("float", [None,10],name='y_') #定义正确的输出值 占位符 类型为float 宽度为10
W_conv1 = weight_variable([5, 5, 1, 32],name='W_conv1')
b_conv1 = bias_variable([32],name='b_conv1')
x_image = tf.reshape(x, [-1,28,28,1],name='x_image')
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1,name='h_conv1')
h_pool1 = max_pool_2x2(h_conv1,name='h_pool1')

tf.summary.histogram("weights", W)
tf.summary.histogram("biases", b)

W_conv2 = weight_variable([5, 5, 32, 64],name='W_conv2')
b_conv2 = bias_variable([64],name='b_conv2')

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2,name='h_conv2')
h_pool2 = max_pool_2x2(h_conv2,name='h_pool2')

W_fc1 = weight_variable([7 * 7 * 64, 1024],name='W_fc1')
b_fc1 = bias_variable([1024],name='b_fc1')

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64],name='h_pool2_flat')
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1,name='h_fc1')

keep_prob = tf.placeholder(tf.float32,name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob,name='h_fc1_drop')

W_fc2 = weight_variable([1024, 10],name='w_fc2')
b_fc2 = bias_variable([10],name='b_fc2')

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,name='y_conv')

#cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    tf.summary.scalar("cross_entropy", cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

sess = tf.Session()
#sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()

def crop_images(images):
     cropedImages = []
     for image in images:
         imgs = ih.crop_Image(ih,image=image)
         for img in imgs:
            cropedImages.append(img)
     return cropedImages
#print("test accuracy {accuracy}".format(accuracy=accuracy.eval(feed_dict={
#    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
def tensorFlowTest():
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, r".\net\sess.ckpt") #读取
    get_result = y_conv
    rawImages = ih.read_images(50,100)
    images = ih.Image_binaryzation(rawImages)
    images = crop_images(images)
    images = ih.resize_images(images)
    images = ih.getdata(images)
    images = ih.normalalizeImages(images)
    names = []
    for eachImg in images :
        testedResult = sess.run(get_result,feed_dict={x:[eachImg],keep_prob: 0.5})
        names.append(max(enumerate(testedResult[0].tolist()),key=lambda x:x[1])[0])
    index = 0
    for img in rawImages:
        img.save(r'.\testOutput\{name}.png'.format(name=''.join(map(str,(names[index * 4:(index + 1) * 4])))))
        index+=1
    print('test finished')

def trainning():
    sess.run(tf.global_variables_initializer())
    merged_summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(r'./tmp/mnist_logs', sess.graph)
    Images = ih.get_trainning_images()
    for i in range(2000):
        ##batch = mnist.train.next_batch(50)
        tempInputList = random.sample(Images,50)#读取数据集
        inX, inY = zip(*tempInputList) #数据集随机切片
        sess.run(train_step,feed_dict={x: inX, y_: inY, keep_prob: 0.5})
        #sess.run(training_op)
        if i % 50 == 0:
            [train_accuracy, s] = sess.run([accuracy, merged_summary_op], feed_dict={x: inX, y_: inY,keep_prob: 1.0})
            writer.add_summary(s, i)
            #train_accuracy = accuracy.eval(feed_dict={x:inX, y_: inY, keep_prob: 1.0},session=sess)
            print("step {i}, training accuracy {accuracy}".format(i=i, accuracy=train_accuracy))
            #summary_str = sess.run(merged_summary_op)
            #writer.add_summary(summary_str, i)
        
    saver = tf.train.Saver()
    saver.save(sess, r".\net\sess.ckpt")
    print('trainning finished')

def main():
    #trainning()
    tensorFlowTest()
if __name__ == "__main__":
    main()
    input()
