# coding: utf-8

import tensorflow as tf


class IMGCNNConfig(object):
    """CNN配置参数"""
    require_improvement = 2000 # 如果超过2000轮未提升，提前结束训练
    print_per_batch = 10  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    num_epochs = 10000  # 总迭代轮次
    batch_size = 10  # 每批训练大小

    learning_rate = 1e-3  # 学习率
    dropout_keep_prob = 0.7  # dropout保留比例

    num_classes = 9  # 类别数

    img_w = 1024 # 宽
    img_h = 1024 # 高
    img_c = 3 # 通道


class VGG_16(object):
    """
    VGG-16: 21 - 5(池化层) = 16
    均采用 3*3 卷积核，SAME方式
    输入为2的n次方大小图片

    http://CNSHF-RPA-AI:6006
    """
    def __init__(self, config):
        self.image_height = config.img_h
        self.image_width = config.img_w
        self.image_channel = config.img_c
        self.num_class = config.num_classes
        self.learning_rate = config.learning_rate

        self.build_model()
        pass

    def build_model(self):

        # 第零层：输入层 1024*512*3
        self.input_x = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, self.image_channel], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_class], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.device('/cpu:0'):
            # 一: 两次卷积，一次池化；3*3*64
            with tf.variable_scope("conv1"):
                conv1_weights = tf.get_variable(name="conv1_w", shape=[3, 3, 3, 64], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                conv1_bias = tf.get_variable(name="conv1_b", shape=[64], initializer=tf.constant_initializer(value=0.1))
                conv1 = tf.nn.conv2d(input=self.input_x, filter=conv1_weights, strides=[1, 1, 1, 1], padding="SAME")
                conv1 = tf.nn.bias_add(value=conv1, bias=conv1_bias)
                relu1 = tf.nn.relu(features=conv1)
            with tf.variable_scope("conv2"):
                conv2_weights = tf.get_variable(name="conv2_w", shape=[3, 3, 64, 64], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                conv2_bias = tf.get_variable(name="conv2_b", shape=[64], initializer=tf.constant_initializer(value=0.1))
                conv2 = tf.nn.conv2d(input=relu1, filter=conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
                conv2 = tf.nn.bias_add(value=conv2, bias=conv2_bias)
                relu2 = tf.nn.relu(features=conv2)
            with tf.variable_scope("pool3"):
                pool3 = tf.nn.max_pool(value=relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='pool3')

            # 二：两次卷积，一次池化；3*3*128
            with tf.variable_scope("conv4"):
                conv4_weights = tf.get_variable(name="conv4_w", shape=[3, 3, 64, 128], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                conv4_bias = tf.get_variable(name="conv4_b", shape=[128], initializer=tf.constant_initializer(value=0.1))
                conv4 = tf.nn.conv2d(input=pool3, filter=conv4_weights, strides=[1, 1, 1, 1], padding="SAME")
                conv4 = tf.nn.bias_add(value=conv4, bias=conv4_bias)
                relu4 = tf.nn.relu(features=conv4)
            with tf.variable_scope("conv5"):
                conv5_weights = tf.get_variable(name="conv5_w", shape=[3, 3, 128, 128], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                conv5_bias = tf.get_variable(name="conv5_b", shape=[128], initializer=tf.constant_initializer(value=0.1))
                conv5 = tf.nn.conv2d(input=relu4, filter=conv5_weights, strides=[1, 1, 1, 1], padding="SAME")
                conv5 = tf.nn.bias_add(value=conv5, bias=conv5_bias)
                relu5 = tf.nn.relu(features=conv5)
            with tf.variable_scope("pool6"):
                pool6 = tf.nn.max_pool(value=relu5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='pool6')

            # 三：三次卷积，一次池化；3*3*256
            with tf.variable_scope("conv7"):
                conv7_weights = tf.get_variable(name="conv7_w", shape=[3, 3, 128, 256], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                conv7_bias = tf.get_variable(name="conv7_b", shape=[256], initializer=tf.constant_initializer(value=0.1))
                conv7 = tf.nn.conv2d(input=pool6, filter=conv7_weights, strides=[1, 1, 1, 1], padding="SAME")
                conv7 = tf.nn.bias_add(value=conv7, bias=conv7_bias)
                relu7 = tf.nn.relu(features=conv7)
            with tf.variable_scope("conv8"):
                conv8_weights = tf.get_variable(name="conv8_w", shape=[3, 3, 256, 256], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                conv8_bias = tf.get_variable(name="conv8_b", shape=[256], initializer=tf.constant_initializer(value=0.1))
                conv8 = tf.nn.conv2d(input=relu7, filter=conv8_weights, strides=[1, 1, 1, 1], padding="SAME")
                conv8 = tf.nn.bias_add(value=conv8, bias=conv8_bias)
                relu8 = tf.nn.relu(features=conv8)
            with tf.variable_scope("conv9"):
                conv9_weights = tf.get_variable(name="conv9_w", shape=[3, 3, 256, 256], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                conv9_bias = tf.get_variable(name="conv9_b", shape=[256], initializer=tf.constant_initializer(value=0.1))
                conv9 = tf.nn.conv2d(input=relu8, filter=conv9_weights, strides=[1, 1, 1, 1], padding="SAME")
                conv9 = tf.nn.bias_add(value=conv9, bias=conv9_bias)
                relu9 = tf.nn.relu(features=conv9)
            with tf.variable_scope("pool10"):
                pool10 = tf.nn.max_pool(value=relu9, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='pool10')

            # 四：三次卷积，一次池化；3*3*512
            with tf.variable_scope("conv11"):
                conv11_weights = tf.get_variable(name="conv11_w", shape=[3, 3, 256, 512], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                conv11_bias = tf.get_variable(name="conv11_b", shape=[512], initializer=tf.constant_initializer(value=0.1))
                conv11 = tf.nn.conv2d(input=pool10, filter=conv11_weights, strides=[1, 1, 1, 1], padding="SAME")
                conv11 = tf.nn.bias_add(value=conv11, bias=conv11_bias)
                relu11 = tf.nn.relu(features=conv11)
            with tf.variable_scope("conv12"):
                conv12_weights = tf.get_variable(name="conv12_w", shape=[3, 3, 512, 512], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                conv12_bias = tf.get_variable(name="conv12_b", shape=[512], initializer=tf.constant_initializer(value=0.1))
                conv12 = tf.nn.conv2d(input=relu11, filter=conv12_weights, strides=[1, 1, 1, 1], padding="SAME")
                conv12 = tf.nn.bias_add(value=conv12, bias=conv12_bias)
                relu12 = tf.nn.relu(features=conv12)
            with tf.variable_scope("conv13"):
                conv13_weights = tf.get_variable(name="conv13_w", shape=[3, 3, 512, 512], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                conv13_bias = tf.get_variable(name="conv13_b", shape=[512], initializer=tf.constant_initializer(value=0.1))
                conv13 = tf.nn.conv2d(input=relu12, filter=conv13_weights, strides=[1, 1, 1, 1], padding="SAME")
                conv13 = tf.nn.bias_add(value=conv13, bias=conv13_bias)
                relu13 = tf.nn.relu(features=conv13)
            with tf.variable_scope("pool14"):
                pool14 = tf.nn.max_pool(value=relu13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='pool14')

            # 五：三次卷积，一次池化；3*3*512
            with tf.variable_scope("conv15"):
                conv15_weights = tf.get_variable(name="conv15_w", shape=[3, 3, 512, 512], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                conv15_bias = tf.get_variable(name="conv15_b", shape=[512], initializer=tf.constant_initializer(value=0.1))
                conv15 = tf.nn.conv2d(input=pool14, filter=conv15_weights, strides=[1, 1, 1, 1], padding="SAME")
                conv15 = tf.nn.bias_add(value=conv15, bias=conv15_bias)
                relu15 = tf.nn.relu(features=conv15)
            with tf.variable_scope("conv16"):
                conv16_weights = tf.get_variable(name="conv16_w", shape=[3, 3, 512, 512], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                conv16_bias = tf.get_variable(name="conv16_b", shape=[512], initializer=tf.constant_initializer(value=0.1))
                conv16 = tf.nn.conv2d(input=relu15, filter=conv16_weights, strides=[1, 2, 2, 1], padding="SAME")
                conv16 = tf.nn.bias_add(value=conv16, bias=conv16_bias)
                relu16 = tf.nn.relu(features=conv16)
            with tf.variable_scope("conv17"):
                conv17_weights = tf.get_variable(name="conv17_w", shape=[3, 3, 512, 512], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                conv17_bias = tf.get_variable(name="conv17_b", shape=[512], initializer=tf.constant_initializer(value=0.1))
                conv17 = tf.nn.conv2d(input=relu16, filter=conv17_weights, strides=[1, 2, 2, 1], padding="SAME")
                conv17 = tf.nn.bias_add(value=conv17, bias=conv17_bias)
                relu17 = tf.nn.relu(features=conv17)
            with tf.variable_scope("pool18"):
                # 32*16*512 (上面两层步长改为2后 8*4*512)
                pool18 = tf.nn.max_pool(value=relu17, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='pool18')

            # 以上为标准VGG结构，共五段；以上配置为VGG-16卷积层的配置  114048个参数（未加上偏置）


            # 拉伸
            concat = tf.layers.flatten(pool18)
            fc_input_dim = concat.shape[1]

            # 六：两次全连接，一次softmax
            with tf.variable_scope("fc19"): # 32*16*512*2048=33554432 个参数
                fc19_w = tf.get_variable(name="fc19_w", shape=[fc_input_dim, 2048], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                fc19_b = tf.get_variable(name="fc19_b", shape=[2048], initializer=tf.constant_initializer(value=0.1))
                # fc19 = tf.nn.relu(features=(tf.matmul(a=concat, b=fc19_w) + fc19_b))
                fc19 = tf.nn.relu_layer(concat, fc19_w, fc19_b, name='fc19')
                fc19_drop = tf.nn.dropout(fc19, keep_prob=self.keep_prob, name='fc19_drop')

            with tf.variable_scope("fc20"): # 2048 * 1024 = 2097152 个参数
                fc20_w = tf.get_variable(name="fc20_w", shape=[2048, 1024], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                fc20_b = tf.get_variable(name="fc20_b", shape=[1024], initializer=tf.constant_initializer(value=0.1))
                # fc20 = tf.nn.relu(features=(tf.matmul(a=concat, b=fc20_w) + fc20_b))
                fc20 = tf.nn.relu_layer(fc19_drop, fc20_w, fc20_b, name='fc20')
                fc20_drop = tf.nn.dropout(fc20, keep_prob=self.keep_prob, name='fc20_drop')

            # softmax
            with tf.variable_scope("softmax"): # 1024 * 122 = 124928 个参数
                logits_w = tf.get_variable(name="logits_w", shape=[1024, self.num_class], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                logits_b = tf.get_variable(name="logits_b", shape=[self.num_class], initializer=tf.constant_initializer(value=0.1))
                logits = tf.matmul(a=fc20_drop, b=logits_w) + logits_b
                # 得分列表
                self.scores = tf.nn.softmax(logits=logits, name='scores')
                # 预测类别
                self.y_pred = tf.arg_max(self.scores, 1, name='y_pred')

            with tf.name_scope('optimize'):
                # 损失函数，交叉熵；这里可以用二进制交叉熵损失函数，实现多分类
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y)
                # 按批次求均值不指定参数是对所有值求均值
                # tf.reduce_mean(cross_entropy,0)按行求均值
                # tf.reduce_mean(cross_entropy,1)按列求均值
                self.loss = tf.reduce_mean(cross_entropy)
                # 优化器
                self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            with tf.name_scope("accuracy"):
                # 准确率
                input_cls = tf.argmax(self.input_y, 1)
                correct_pred = tf.equal(input_cls, self.y_pred) # 对比两个数组中对应位置是否相同，输出True，False矩阵
                # 将 x 的数据格式转化成 dtype.
                cs_data = tf.cast(correct_pred, tf.float32) #将True，False矩阵转换成 1， 0 矩阵
                self.acc = tf.reduce_mean(cs_data) # 计算1占矩阵元素个数的百分比





class LeNet_5(object):
    """
    LeNet-5经典五层结构

    http://CNSHF-RPA-AI:6006
    """
    def __init__(self, config):
        self.image_height = config.img_h
        self.image_width = config.img_w
        self.image_channel = config.img_c
        self.num_class = config.num_classes
        self.learning_rate = config.learning_rate

        self.build_model()
        pass

    def build_model(self):
        # 第零层：输入层 32*32*3
        self.input_x = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, self.image_channel], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_class], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.device('/cpu:0'):

            # wide = (224 + 2 * padding - kernel_size) / stride + 1

            # 第一层：卷积层
            # 卷积核：5*5*6
            # 输出： 28*28*6
            with tf.variable_scope("conv1"):
                conv1_weights = tf.get_variable(name="conv1_w", shape=[5, 5, 3, 6], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                conv1_bias = tf.get_variable(name="conv1_b", shape=[6], initializer=tf.constant_initializer(value=0.1))
                conv1 = tf.nn.conv2d(input=self.input_x, filter=conv1_weights, strides=[1, 1, 1, 1], padding="VALID")
                conv1 = tf.nn.bias_add(value=conv1, bias=conv1_bias)
                relu1 = tf.nn.relu(features=conv1)

            # 第二层：池化
            # 步长：1*2*2*1
            # 输出：14*14*6
            with tf.variable_scope("pool2"):
                pool2 = tf.nn.avg_pool(value=relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
                # tf.nn.max_pool()
                # pooling = tf.layers.max_pooling2d(relu1, [2, 2], [2, 2], name='pooling')

            # 第三层：卷积层
            # 卷积核：5*5*16
            # 输出：10*10*16
            with tf.variable_scope("conv3"):
                conv3_weights = tf.get_variable(name="conv3_w", shape=[5, 5, 6, 16], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                conv3_bias = tf.get_variable(name="conv3_b", shape=[16], initializer=tf.constant_initializer(value=0.1))
                conv3 = tf.nn.conv2d(input=pool2, filter=conv3_weights, strides=[1, 1, 1, 1], padding="VALID")
                conv3 = tf.nn.bias_add(value=conv3, bias=conv3_bias)
                relu3 = tf.nn.relu(features=conv3)

            # 第四层：池化层
            # 步长：1*2*2*1
            # 输出：5*5*16
            with tf.variable_scope("pool4"):
                pool4 = tf.nn.avg_pool(value=relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

            # 第五层：卷积
            # 卷积核：5*5*120
            # 输出：1*1*120
            with tf.variable_scope("conv5"):
                conv5_weights = tf.get_variable(name="conv5_w", shape=[5, 5, 16, 120], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                conv5_bias = tf.get_variable(name="conv5_b", shape=[120], initializer=tf.constant_initializer(value=0.1))
                conv5 = tf.nn.conv2d(input=pool4, filter=conv5_weights, strides=[1, 1, 1, 1], padding="VALID")
                conv5 = tf.nn.bias_add(value=conv5, bias=conv5_bias)
                relu5 = tf.nn.relu(features=conv5)
            # 拉伸
            # shape_of_feature_map = relu5.get_shape().as_list()
            # fc_input = tf.reshape(tensor=relu5, shape=[batch_size, len_feature_map])
            concat = tf.layers.flatten(relu5)
            fc_input_dim = concat.shape[1]

            # 第六层：全连接层
            # 神经元数：84
            # 参数个数：120 * 84
            # 输出：84
            with tf.variable_scope("fc6"):
                fc6_weights = tf.get_variable(name="w6", shape=[fc_input_dim, 84], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                fc6_bias = tf.get_variable(name="b6", shape=[84], initializer=tf.constant_initializer(value=0.1))
                fc6 = tf.nn.relu(features=(tf.matmul(a=concat, b=fc6_weights) + fc6_bias))

            # 第七层：Gaussian Connection输出层
            # 神经元数：类别数
            # 参数个数：84 * 类别数
            # 输出：类别数
            with tf.variable_scope("softmax"):
                fc7_weights = tf.get_variable(name="w7", shape=[84, self.num_class], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                fc7_bias = tf.get_variable(name="b7", shape=[self.num_class], initializer=tf.constant_initializer(value=0.1))
                logits = tf.matmul(a=fc6, b=fc7_weights) + fc7_bias
                # 得分列表
                self.scores = tf.nn.softmax(logits=logits, name='scores')
                # 预测类别
                self.y_pred = tf.arg_max(self.scores, 1, name='y_pred')

            with tf.name_scope('optimize'):
                # 损失函数，交叉熵；这里可以用二进制交叉熵损失函数，实现多分类
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y)
                # 按批次求均值不指定参数是对所有值求均值
                # tf.reduce_mean(cross_entropy,0)按行求均值
                # tf.reduce_mean(cross_entropy,1)按列求均值
                self.loss = tf.reduce_mean(cross_entropy)
                # 优化器
                self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            with tf.name_scope("accuracy"):
                # 准确率
                input_cls = tf.argmax(self.input_y, 1)
                correct_pred = tf.equal(input_cls, self.y_pred) # 对比两个数组中对应位置是否相同，输出True，False矩阵
                # 将 x 的数据格式转化成 dtype.
                cs_data = tf.cast(correct_pred, tf.float64) #将True，False矩阵转换成 1， 0 矩阵
                self.acc = tf.reduce_mean(cs_data) # 计算1占矩阵元素个数的百分比




# class IMG_CNN(object):
#     def __init__(self, config):
#         self.config = config
#
#         # 输入图像(归一化灰度值)：b * h * w * c;
#         self.input_x = tf.placeholder(tf.float32, [None, config.img_h, config.img_w, config.img_c], name='input_x')
#         self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
#         self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
#
#         self.cnn()
#
#     def cnn(self):
#         """改为AlexNet 或 LeNet-5"""
#         with tf.name_scope("cnn"):
#             # 卷积层 b * w * h * c --> b * wc * hc * f
#             conv = tf.layers.conv2d(self.input_x,
#                                     filters=self.config.num_filters,
#                                     kernel_size=self.config.kernel_size,
#                                     strides=(1, 1),
#                                     # 在same情况下，只有在步长为1时生成的feature map才会和输入值相等
#                                     padding='valid',
#                                     activation=None,
#                                     use_bias=True,
#                                     name='conv')
#             # 池化层
#             pooling = tf.layers.max_pooling2d(conv, [2, 2], [2, 2], name='pooling')
#             # concat
#             concat = tf.layers.flatten(pooling)
#
#         with tf.name_scope("score"):
#             # 全连接层，后面接dropout以及relu激活
#             fc = tf.layers.dense(concat, self.config.hidden_dim, name='fc1')
#             fc = tf.contrib.layers.dropout(fc, self.keep_prob)
#             fc = tf.nn.relu(fc)
#
#             # 分类器
#             self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
#             self.y_pred_score = tf.nn.softmax(self.logits, name='y_pred_score')
#             self.y_pred_cls = tf.argmax(self.y_pred_score, 1, name='y_pred_cls')  # 预测类别
#
#         with tf.name_scope("optimize"):
#             # 损失函数，交叉熵
#             cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
#             self.loss = tf.reduce_mean(cross_entropy)
#             # 优化器
#             self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
#
#         with tf.name_scope("accuracy"):
#             # 准确率
#             correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
#             self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


if __name__ == '__main__':
    # """
    # 经典五层结构
    # input:      32*32*3
    # conv_1:     28*28*6 (5*5*6)
    # pooling_1:  14*14*6 (1*2*2*1)
    # conv_2:     10*10*16 (5*5*16)
    # pooling_2:  5*5*16 (1*2*2*1)
    # conv_3:     1*1*120    (5*5*120)
    # fc:         84 (权重120 * 84)
    # output:     num_class
    #
    # 特征工程：
    # 1、归一化图像尺寸
    # 2、二值化图像
    # """
    #
    #
    # # model_config = IMGCNNConfig()
    # # model = IMG_CNN(model_config)
    #
    # # input = tf.Variable(tf.random_normal([1, 5, 5, 3]))
    # # filter = tf.Variable(tf.random_normal([3, 3, 3, 7]))
    # # # 需要自己初始化权重
    # # # op6 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
    # #
    # # # 不需要初始化权重
    # # op6 = tf.layers.conv2d(input,
    # #                         filters=7,
    # #                         kernel_size=3,
    # #                         strides=(1, 1),
    # #                         # 在same情况下，只有在步长为1时生成的feature map才会和输入值相等
    # #                         padding='SAME',
    # #                         activation=None,
    # #                         use_bias=True,
    # #                         name='conv')
    # #
    # #
    #
    # input_x = tf.Variable(tf.random_normal([1, 7, 7, 1]))
    # input_x = tf.Print(input_x, data=["INPUT:", input_x.shape, "----", input_x], summarize=100)
    #
    # conv1_weights = tf.get_variable(name="conv1_w", shape=[3, 3, 1, 2], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
    # conv1_weights = tf.Print(conv1_weights, data=["FILTER:", conv1_weights.shape, "----", conv1_weights], summarize=100)
    #
    # conv1_bias = tf.get_variable(name="conv1_b", shape=[2], initializer=tf.constant_initializer(value=0.1))
    # conv1_bias = tf.Print(conv1_bias, data=["BIAS:", conv1_bias.shape, "----", conv1_bias], summarize=100)
    #
    # conv1 = tf.nn.conv2d(input=input_x, filter=conv1_weights, strides=[1, 1, 1, 1], padding="VALID")
    # conv1 = tf.Print(conv1, data=["FM-CONV:", conv1.shape, "----", conv1], summarize=100)
    #
    # conv1_add_bias = tf.nn.bias_add(value=conv1, bias=conv1_bias)
    # conv1_add_bias = tf.Print(conv1_add_bias, data=["FM-BIAS:", conv1_add_bias.shape, "----", conv1_add_bias], summarize=100)
    #
    # relu1 = tf.nn.relu(features=conv1_add_bias)
    # relu1 = tf.Print(relu1, data=["FM-RELU:", relu1.shape, "----", relu1], summarize=100)
    #
    # concat = tf.layers.flatten(relu1)
    # concat = tf.Print(concat, data=["CONCAT:", concat.shape, "----", concat], summarize=100)
    #
    # # concat_ = tf.reshape(tensor=relu1, shape=[1, -1])
    # # concat_ = tf.Print(concat_, data=["CONCAT:", concat_.shape, "----", concat_], summarize=100)
    #
    # with tf.Session() as sess:
    #     init = tf.initialize_all_variables()
    #     sess.run(init)
    #     print_tf = sess.run([concat])
    #
    #
    #
    #
    # # config = IMGCNNConfig()
    # # model = LeNet_5(config)

    # VGG_16(IMGCNNConfig())

    # with tf.Session() as sess:
    #     input_x = tf.placeholder(tf.float32, [32, 2016, 672, 3], name='input_x')
    #
    #     conv1_weights = tf.get_variable(name="conv1_w", shape=[11, 11, 3, 64], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
    #     conv1_bias = tf.get_variable(name="conv1_b", shape=[64], initializer=tf.constant_initializer(value=0.1))
    #     conv1 = tf.nn.conv2d(input=input_x, filter=conv1_weights, strides=[1, 2, 2, 1], padding="VALID")
    #     conv1 = tf.nn.bias_add(value=conv1, bias=conv1_bias)
    #     relu1 = tf.nn.relu(features=conv1)
    #
    #     pool2 = tf.nn.max_pool(value=relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID") # .avg_pool(value=relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    #
    #     conv3_weights = tf.get_variable(name="conv3_w", shape=[5, 5, 64, 128], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
    #     conv3_bias = tf.get_variable(name="conv3_b", shape=[128], initializer=tf.constant_initializer(value=0.1))
    #     conv3 = tf.nn.conv2d(input=pool2, filter=conv3_weights, strides=[1, 2, 2, 1], padding="VALID")
    #     conv3 = tf.nn.bias_add(value=conv3, bias=conv3_bias)
    #     relu3 = tf.nn.relu(features=conv3)
    #
    #     pool4 = tf.nn.max_pool(value=relu3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
    #
    #     print("")

    # LeNet_5(IMGCNNConfig())
    pass
