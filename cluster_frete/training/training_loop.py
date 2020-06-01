import tensorflow as tf
from cluster_frete.classifier_model import ClassifierModel


def sigmoid_loss(logits, labels):
    return tf.compat.v1.losses.sigmoid_cross_entropy(labels, logits)


def training_step(model, features, labels, optimizer):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        predicted = model(features, training=True)

        loss = sigmoid_loss(predicted, labels)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    model.last_step.assign_add(1)
    model.loss.assign(loss)
    model.accuracy.update_state(predicted, labels)


def train(model: ClassifierModel, dataset: tf.data.Dataset, save_path: str, summary_step: int, save_step: int, optimizer):

    for features, label in dataset:
        training_step(model, features, label, optimizer)

        last_step = model.last_step.value()

        if last_step % summary_step == 0:
            model.summary()

        if last_step % save_step == 0:
            model.save(save_path)
