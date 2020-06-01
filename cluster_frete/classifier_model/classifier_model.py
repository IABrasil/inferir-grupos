import tensorflow as tf

from cluster_frete.classifier_model.model_utils import create_inner_model


class ClassifierModel(tf.Module):

    def __init__(self, layer_sizes, name=None):
        super().__init__(name=name)
        self.inner_model = create_inner_model(layer_sizes)
        self._initialize_metrics()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 5], dtype=tf.float32),
                                  tf.TensorSpec(shape=(), dtype=tf.bool)])
    def __call__(self, features, training=False):
        return self.inner_model(features, training=training)

    def _initialize_metrics(self):
        self.last_step: tf.Variable = tf.Variable(0, dtype=tf.int32, name="last_step", trainable=False)
        self.loss: tf.Variable = tf.Variable(0.0, dtype=tf.float32, name="loss", trainable=False)
        self.accuracy = tf.keras.metrics.BinaryAccuracy(dtype=tf.float32, name='accuracy')

    def save(self, save_path):
        tf.saved_model.save(self, save_path)

    def summary(self):
        accuracy = self.accuracy.result().numpy()
        tf.print(self.last_step.value(), self.loss.value(), accuracy, sep=",")
