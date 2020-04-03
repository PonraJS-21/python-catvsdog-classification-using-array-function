import numpy as np
import os
import cv2
from keras.models import model_from_json
import pickle



classifier_f = open("int_to_word_out.pickle", "rb")
int_to_word_out = pickle.load(classifier_f)
classifier_f.close()



# load json and create model
json_file = open('model_face.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_face.h5")
print("Model is now loaded in the disk")

path = 'predict/Dog/'

img=os.listdir(path)[1]
image=np.array(cv2.imread(path+img))
image = cv2.resize(image, (150, 150))
image=np.array([image])
image = image.astype('float32')
image = image / 255.0

prediction=loaded_model.predict(image)

print(prediction)

print(np.max(prediction))

print(int_to_word_out[np.argmax(prediction)])
