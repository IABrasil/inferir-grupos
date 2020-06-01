import os

import tensorflow as tf

COLUMN_NAMES = ["frete", "preco", "prazo", "latitude", "longitude", "grupo1", "grupo0"]

def _prepare_supervised(row):
    frete = row["frete"]
    preco = row["preco"]
    prazo = tf.cast(row["prazo"], tf.float32)
    latitude = row["latitude"]
    longitude = row["longitude"]

    grupo1 = tf.cast(row["grupo1"], tf.float32)
    grupo0 = tf.cast(row["grupo0"], tf.float32)

    label = tf.stack([grupo1, grupo0], axis=1)

    return tf.stack([frete, preco, prazo, latitude, longitude], axis=1), label


def load_dataset_for_supervised(source_path, batch_size, repeats):
    file_pattern = os.path.join(source_path, "part*")
    buffer_size = 100

    return tf.data.experimental\
            .make_csv_dataset(file_pattern, batch_size,
                              shuffle_buffer_size=buffer_size,
                              prefetch_buffer_size=buffer_size,
                              num_parallel_reads=tf.data.experimental.AUTOTUNE,
                              column_names=COLUMN_NAMES)\
            .map(_prepare_supervised)\
            .repeat(repeats)


def _prepare_unsupervised(row):
    frete = row["frete"]
    preco = row["preco"]
    prazo = tf.cast(row["prazo"], tf.float32)
    latitude = row["latitude"]
    longitude = row["longitude"]

    label = tf.cast(row["grupo1"], tf.int32)

    return tf.stack([frete, preco, prazo, latitude, longitude], axis=1), label


def load_dataset_for_unsupervised(source_path, batch_size, repeats):
    file_pattern = os.path.join(source_path, "part*")
    buffer_size = 100

    return tf.data.experimental\
            .make_csv_dataset(file_pattern, batch_size,
                              shuffle_buffer_size=buffer_size,
                              prefetch_buffer_size=buffer_size,
                              num_parallel_reads=tf.data.experimental.AUTOTUNE,
                              column_names=COLUMN_NAMES)\
            .map(_prepare_unsupervised)\
            .repeat(repeats)
