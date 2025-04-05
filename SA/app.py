from flask import Flask, render_template, request
import numpy as np
import re
import os
import tensorflow as tf
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model

# Define the folder path for images
IMAGE_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)

# Configure the image folder
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

def init():
    global model, graph
    # Load the pre-trained Keras model
    model = load_model('sentiment_analysis.h5')
    graph = tf.get_default_graph()

# Home route - display the form
@app.route('/')
def home():
    return render_template("home.html")

# Sentiment analysis route - process the form submission
@app.route('/sentiment_analysis_prediction', methods=['POST'])
def sent_anly_prediction():
    if request.method == 'POST':
        # Get text from the form
        text = request.form['text']
        
        # Process the text for prediction
        max_review_length = 500
        word_to_id = imdb.get_word_index()
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        text = text.lower().replace("<br />", " ")
        text = re.sub(strip_special_chars, "", text.lower())

        # Convert text to sequence
        words = text.split()
        x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word] <= 20000) else 0 for word in words]]
        x_test = sequence.pad_sequences(x_test, maxlen=500)  # Should match training data length
        vector = np.array([x_test.flatten()])
        
        # Make prediction
        with graph.as_default():
            probability = model.predict(array([vector][0]))[0][0]
            class1 = model.predict_classes(array([vector][0]))[0][0]
        
        # Determine sentiment based on prediction
        if class1 == 0:
            sentiment = 'Negative'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Sad_Emoji.png')
        else:
            sentiment = 'Positive'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Smiling_Emoji.png')
        
        # Render template with results
        return render_template('home.html', text=text, sentiment=sentiment, 
                              probability=probability, image=img_filename)
    
    # If not POST, redirect to home
    return render_template('home.html')

if __name__ == "__main__":
    init()
    app.run(debug=True)