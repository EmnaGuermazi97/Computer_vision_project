import cv2
import tensorflow as tf
import keras.callbacks
CATEGORIES = ["fastGEI", "normalGEI", "slowGEI"]
def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
model = tf.keras.models.load_model("CNN.model")
image = "C:/Users/lenovo/Desktop/projet vision final/dataGEI/fastGEI/gei_fq01.png" #your image path
predictions = model.predict(image, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)
predictions[0]