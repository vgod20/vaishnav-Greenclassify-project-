from flask import Flask, render_template, request
import tensorflow as tf
from keras.utils import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model('vegetable_model.h5')

labels = [
    'Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
    'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
    'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['file']

    if not os.path.exists('static'):
        os.makedirs('static')

    img_path = os.path.join('static', img_file.filename)
    img_file.save(img_path)

    # Match training size
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = labels[np.argmax(prediction)]

    return f"<h2>Result: {result}</h2><a href='/'>Back</a>"

if __name__ == '__main__':
    app.run(debug=False)
