import argparse
import tensorflow as tf
import keras

from models import get_model

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, help="name of the model")
parser.add_argument("--img_size", type=int, required=True, help="size of each image dimension")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
opt = parser.parse_args()


def train_model(model: keras.Model, train_ds, val_ds, epochs):
    model.fit(train_ds, epochs=epochs, validation_data=val_ds)

    print(f"Saving model to: ./SavedModel/{model.name}.h5")
    model.save(f"./SavedModel/{model.name}.h5")


@tf.function
def mapping(img, label, size):
    img = tf.image.resize(img, size)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img / 255

    return img, label


if __name__ == "__main__":
    print(f"Get model {opt.model_name}")
    model = get_model(opt.model_name, 10)
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_ds = train_ds.map(lambda img, label: mapping(img, label, (opt.img_size, opt.img_size)))
    test_ds = test_ds.map(lambda img, label: mapping(img, label, (opt.img_size, opt.img_size)))

    train_ds = train_ds.batch(32).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(32).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)

    print(f"Train model {opt.model_name} on CIFAR-10 dataset")
    train_model(model, train_ds, test_ds, opt.epochs)