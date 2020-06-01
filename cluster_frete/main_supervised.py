import tensorflow as tf

from cluster_frete.classifier_model import ClassifierModel
from cluster_frete.dados import load_dataset_for_supervised
from cluster_frete.training.training_loop import train


def main():
    source_directory = "dados/"
    save_path = "output"
    batch_size = 500
    repeats = 10
    layer_sizes = [20, 10]

    summary_step = 100
    save_step = 100

    learning_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    dataset = load_dataset_for_supervised(source_directory, batch_size, repeats)
    model = ClassifierModel(layer_sizes)

    train(model, dataset, save_path, summary_step, save_step, optimizer)


if __name__ == '__main__':
    main()
