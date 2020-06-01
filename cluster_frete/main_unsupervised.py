import numpy as np
import tensorflow as tf

from cluster_frete.dados.loader import load_dataset_for_unsupervised


def main():
    source_directory = "dados/"
    batch_size = 50000
    num_iterations = 15

    dataset = load_dataset_for_unsupervised(source_directory, batch_size, repeats=1).as_numpy_iterator()
    batch, labels = next(dataset)

    def input_fn():
        return tf.compat.v1.train.limit_epochs(tf.convert_to_tensor(batch, dtype=tf.float32), num_epochs=1)

    num_clusters = 2
    kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=num_clusters, use_mini_batch=False)

    for _ in np.arange(num_iterations):
        kmeans.train(input_fn)

    accuracy = calculate_accuracy(input_fn, kmeans, labels)
    print(accuracy.numpy())


def calculate_accuracy(input_fn, kmeans, labels):
    cluster_indices = tf.constant(list(kmeans.predict_cluster_index(input_fn)))
    cluster_inces_reverted = tf.abs(cluster_indices - 1)
    cluster_indices = cluster_indices
    cluster_indices_reverted = cluster_inces_reverted
    accuracy = tf.keras.metrics.binary_accuracy(cluster_indices, labels)
    accuracy_reverted = tf.keras.metrics.binary_accuracy(cluster_indices_reverted, labels)
    accuracy = accuracy if accuracy >= accuracy_reverted else accuracy_reverted
    return accuracy


if __name__ == '__main__':
    main()