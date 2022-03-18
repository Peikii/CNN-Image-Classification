import numpy as np
import tensorflow as tf

class Model_Init(object):
    def __init__(self, model_dir):
        print("初始化模型...")
        # 获取最新的模型名
        model_save_path = tf.train.latest_checkpoint(model_dir)
        print("latest model '{}'".format(model_save_path))

        self.graph = tf.Graph()
        with self.graph.as_default():
            # 加载图
            self.saver = tf.train.import_meta_graph(model_save_path+".meta")
        self.sess = tf.Session(graph=self.graph)  # 创建新的sess
        # self.saver.restore(self.sess, save_path=model_save_path)  # 从恢复点恢复参数
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess, save_path=model_save_path)  # 从恢复点恢复参数


    def predict(self, input_x):
        feed_dict = {
            "input_x:0": input_x,
            "keep_prob:0": 1.00
        }
        # 或者 op = tf.get_default_graph().get_operation_by_name("ArgMax").outputs[0]  两种方式获取op都可以
        # x = self.graph.get_tensor_by_name("input_x:0")
        # d = self.graph.get_tensor_by_name("keep_prob:0")
        # e = self.graph.get_tensor_by_name("embedding:0")

        op_soft_max = self.graph.get_tensor_by_name("score/Softmax:0")
        op_arg_max = self.graph.get_tensor_by_name("score/ArgMax:0")
        # op = tf.get_default_graph().get_operation_by_name("score/ArgMax").outputs[0]
        score, y_pred = self.sess.run([op_soft_max, op_arg_max], feed_dict=feed_dict)
        # print(ox)
        # print(od)
        # print(oe.shape)
        # print(oe)
        print(score)
        return np.amax(score[0], axis=0), y_pred[0]

    def predict_batch(self, input_x):
        """
        批量预测，主要用于模型评估
        :param input_x:
        :return:
        """
        feed_dict = {
            "input_x:0": input_x,
            "keep_prob:0": 1.0
        }
        op_soft_max = self.graph.get_tensor_by_name("score/Softmax:0")
        op_arg_max = self.graph.get_tensor_by_name("score/ArgMax:0")
        score, y_pred = self.sess.run([op_soft_max, op_arg_max], feed_dict=feed_dict)
        return np.amax(score, axis=1), y_pred


    def predict_img(self, input_x):
        """

        :param input_x:
        :return:
        """
        try:
            feed_dict = {
                "input_x:0": input_x,
                "keep_prob:0": 1.0
            }

            op_soft_max = self.graph.get_tensor_by_name("softmax/scores:0")
            op_arg_max = self.graph.get_tensor_by_name("softmax/y_pred:0")
            score, y_pred = self.sess.run([op_soft_max, op_arg_max], feed_dict=feed_dict)

            return np.amax(score[0], axis=0), y_pred[0]
        except Exception as e:
            raise e

    def predict_img_batch(self, input_x):
        """

        :param input_x:
        :return:
        """
        try:
            feed_dict = {
                "input_x:0": input_x,
                "keep_prob:0": 1.0
            }
            op_soft_max = self.graph.get_tensor_by_name("softmax/scores:0")
            op_arg_max = self.graph.get_tensor_by_name("softmax/y_pred:0")
            score, y_pred = self.sess.run([op_soft_max, op_arg_max], feed_dict=feed_dict)
            return np.amax(score, axis=1), y_pred
        except Exception as e:
            raise e