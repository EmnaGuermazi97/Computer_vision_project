# this version didn't work out
import tensorflow as tf
import keras.callbacks
#from keras.models import load_model

model = tf.keras.models.load_model("CNN.model")
image = "C:/Users/lenovo/Desktop/projet vision final/dataGEI/fastGEI/gei_fq01.png" #your image path

tb_test = keras.callbacks.TensorBoard(log_dir=strPath_model_test_logs,histogram_freq=0, write_graph=True, write_images=True)

y_test = model.predict(image, verbose=1, callbacks=[tb_test])
