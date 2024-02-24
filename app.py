from flask import Flask, request, jsonify
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf



app = Flask(__name__)

model = load_model('my_model.h5')

# Mock function representing your machine learning model
def process_image(image):
    class_names = ['black_pepper_healthy', 'black_pepper_leaf_blight','black_pepper_yellow_mottle_virus']
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 1)


    my_data = predicted_class + "----" + str(confidence) +"%"

    # Your image processing code here
    # This function should take an image as input and return the extracted text
    # For demonstration purposes, let's assume the model is a placeholder function

    extracted_text = my_data
    return extracted_text

# Endpoint to handle POST requests
@app.route('/extract_text', methods=['POST'])
def extract_text():
    # Check if request contains file data
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if the file is an image
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        return jsonify({'error': 'Unsupported file format'}), 400
    
    try:
        # Open the image using PIL (Python Imaging Library)
        image = Image.open(file)

        print (image)
         # Convert the image to a NumPy array
    
        image_array = np.asarray(image)
        new_image_array = np.resize(image_array, (256, 256,3))
        # Process the image using your machine learning model
        extracted_text = process_image(new_image_array)

        
        return jsonify({'text': extracted_text}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
