"""
文本分类
神经网络
"""
import tensorflow as tf

class ModelConfig(object):
    """
    模型参数配置
    tensorboard --logdir=C:\YF\code\pypi\robot_platform\dl_classification\tensorboard\pdf_class
    tensorboard --logdir=/home/irobot/txt classification/dl_classification/tensorboard/pdf_class
    tensorboard --logdir=ed_hold
    tensorboard --logdir=pdf_company

    http://cnsho-3cjt5h2:6006/#scalars


    tensorboard --logdir=ed_hold
     http://CNSHF-RPA-AI:6006
     http://cnshf-rpa-ai:6006/
    """
    def __init__(self):
        self.num_epochs = 10000 # 总迭代次数
        self.batch_size = 32 # 每批训练样本数
        self.seq_length = 4000 # 句子长度
        self.embedding_dim = 64 # 词向量维度
        self.num_class = 2 # 类别数
        self.vocab_size = 5000  # 字典大小

        self.rnn = 'lstm' # 采用模型类别，lstm、gru
        self.rnn_num_layers =2 # rnn层数

        self.rnn_hidden_dim = 128
        self.dense_hidden_dim = 128 # 全连接层，隐藏层的维度

        self.dropout_keep_prob = 0.8 # dropout保留比例,rnn 全连接 公用这个参数
        self.learning_rate = 1e-3 # 学习率 0.001



class CRNNModel(object):
    """模型搭建"""
    def __init__(self, config):
        self.config = config
        # 定义模型输入
        # batch_size * seq_length
        self.input_x = tf.placeholder(name='input_x', shape=[None, self.config.seq_length], dtype=tf.int32)
        # batch_size * num_class
        self.input_y = tf.placeholder(name='input_y', shape=[None, self.config.num_class], dtype=tf.float32)
        self.keep_prob = tf.placeholder(name='keep_prob', dtype=tf.float32)

        self.crnn()

    def crnn(self):
        """搭建网络"""
        def lstm_cell(): # lstm 细胞
            return tf.contrib.rnn.BasicLSTMCell(self.config.rnn_hidden_dim, state_is_tuple=True)
        def gru_cell(): # GRU 层
            return tf.contrib.rnn.GRUCell(self.config.rnn_hidden_dim)
        def dropout():
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            # 暂时公用 dropout 吧。
            dw = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            return dw

        with tf.device("/cpu:0"):
            # 读入字典 词数 * 词向量维度。
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            # 给每批样本嵌入词向量，batch_size * 600 * 64
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope('rnn'):
            # 创建2个rnn
            cells = [dropout() for _ in range(self.config.rnn_num_layers)]
            # 构建两层rnn网络，state_is_tuple = True是把lstm时间步长上状态集合c，和输出h组装成一个元组（c,h）
            multiRNNCell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            # 执行构建好的多层rnn，相当于向前传播
            # 同一个batch里的句子必须等长，要进行padding，不同batch可以不同。如果这里没有指定参数sequemce_length=，会自动获取inputs中句子长度，也就是embedding_inputs[1]的长度
            # 返回的是元组（每一步的输出h=rnn层数*batch*seqlen*词维度, 隐状态c=rnn层数*batch*1*词维度）
            rnn_outputs_h,  rnn_outputs_c= tf.nn.dynamic_rnn(cell=multiRNNCell, inputs=embedding_inputs, dtype=tf.float32)
            # 取rnn最后一个时序的输出作为句子特征。这里取所有时序输出做其他特征处理
            last_h_output = rnn_outputs_h[:,-1 ,:] # batchsize * 128

        with tf.name_scope('score'):
            # 全连接层，隐藏层
            fc1 = tf.layers.dense(last_h_output, self.config.dense_hidden_dim, name='fc1')
            fc1_dropout = tf.contrib.layers.dropout(fc1, self.keep_prob)
            fc1_active = tf.nn.relu(fc1_dropout)

            # 全连接层，输出层（分类器）64 * 10
            logits = tf.layers.dense(fc1_active, self.config.num_class, name='fc_ouput')
            l_s = tf.nn.softmax(logits)
            # tf.argmax这里是取概率最大的类别.axis = 0 的时候返回每一列（第一个维度）最大值的位置索引。 1代表第二个维度（列）。多维以此类推
            # 64 * 1
            self.y_pred_cls = tf.arg_max(l_s, 1) # 这里如果是多分类可以取topk，也可以取阈值来取多个分类

        with tf.name_scope('optimize'):
            # 损失函数，交叉熵；这里可以用二进制交叉熵损失函数，实现多分类
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y)
            # 按批次求均值不指定参数是对所有值求均值
            # tf.reduce_mean(cross_entropy,0)按行求均值
            # tf.reduce_mean(cross_entropy,1)按列求均值
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            input_y_cls = tf.argmax(self.input_y, 1)
            correct_pred = tf.equal(input_y_cls, self.y_pred_cls) # 对比两个数组中对应位置是否相同，输出True，False矩阵
            # 将 x 的数据格式转化成 dtype.
            cs_data = tf.cast(correct_pred, tf.float32) #将True，False矩阵转换成 1， 0 矩阵
            self.acc = tf.reduce_mean(cs_data) # 计算1占矩阵元素个数的百分比


if __name__ == '__main__':
    config = ModelConfig()

    config.num_epochs = 20000
    config.batch_size = 64
    config.seq_length = 21
    config.embedding_dim = 64
    config.num_class = 10
    config.vocab_size = 8000
    config.dropout_keep_prob = 0.8
    config.learning_rate = 1e-3
    print(config.__dict__.items())

    model = CRNNModel(config)






















