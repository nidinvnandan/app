import streamlit as st
from PIL import Image
import numpy as np
from numpy import argmax
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle
from sklearn.datasets import load_iris

#loading models for different tasks
try:
    modelcnn1 = tf.keras.models.load_model('cnnmodel.h5')
except Exception as e:
    st.error(f" error loading CNN Model: {e}")
dnnmoviemodel=tf.keras.models.load_model('dnn_movie.h5')
lstmmodel = tf.keras.models.load_model('lstm_imdb_model.h5')
digitmodel=tf.keras.models.load_model('digit.h5')
rnnmodel=tf.keras.models.load_model('model_kerasrnn.h5')
with open('tokeniser.pkl', 'rb') as file:
    loaded_tokeniser = pickle.load(file)
with open('backpropagationmovie_model.pkl','rb') as file:
     BP_model=pickle.load(file)  
with open('perceptronmovie_model.pkl','rb') as file:
     pm_model=pickle.load(file)
with open('dnntokeniser.pkl','rb') as file:
     dtokenizer=pickle.load(file)   

iris=load_iris()
class_names = iris.target_names

#defining functions for the prediction
#make function for cnn tumor prediction
def make(img):
    img = Image.open(img)
    img = img.resize((128, 128))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)
    res = modelcnn1.predict(input_img)
    if res[0][0] >= 0.5:  # Considering threshold for classification
        return "Tumor Detected"
    else:
        return "No Tumor"
    
#function for DNN prediction 
def predict_dnnsentiment(text):
    sequence = dtokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = dnnmoviemodel.predict(padded_sequence)[0][0]
    if prediction >= 0.5:
        return "Positive"
    else:
        return "Negative"


#function to predcit movie review using LSTM model          
def predict_sentiment(review):
    # Process input text similarly to training data
    top_words = 5000
    max_review_length=500
    word_index = imdb.get_word_index()
    review = review.lower().split()
    review = [word_index[word] if word in word_index and word_index[word] < top_words else 0 for word in review]
    review = sequence.pad_sequences([review], maxlen=max_review_length)
    prediction = lstmmodel.predict(review)
    if prediction > 0.5:
        return "Positive"
    else:
        return "Negative"
    

def preprocess_user_image(image):
    # Open the image using PIL
    img = Image.open(image)
    img = img.convert('L')  # Convert to grayscale

    # Resize the image to match MNIST dataset's input shape (28x28 pixels)
    img = img.resize((28, 28))

    # Convert image to array and normalize
    img_array = np.array(img)
    img_array = img_array.reshape((1, 28, 28, 1))  # Reshape to match model input shape
    img_array = img_array.astype('float32') / 255.0  # Normalize pixels to range [0, 1]

    return img_array

#function to predciting the digit using cnn
def predict_digit(image):
    # Preprocess the image (reshape if necessary, normalize, etc.)
    # Make predictions using the loaded model
    image = preprocess_user_image(image)  # Replace preprocess_image with your actual preprocessing steps
    prediction = digitmodel.predict(image)
    predicted_label = argmax(prediction, axis=1)
    return predicted_label[0]


#fuction to predcit spam message using RNN
def rnn_predict(input_text):
    # Process input text similarly to training data
    #here the loaded_tokeniser is a function which is defined already
    encoded_input = loaded_tokeniser.texts_to_sequences([input_text])
    padded_input = tf.keras.preprocessing.sequence.pad_sequences(encoded_input, maxlen=10, padding='post')
    prediction = rnnmodel.predict(padded_input)
    if prediction > 0.5:
        return "spam"
    else:
        return "ham"


#preprocess function for perceptron
def preprocess_input(user_input, num_words=1000, max_len=200):
    word_index = imdb.get_word_index()
    input_sequence = [word_index[word] if word in word_index and word_index[word] < num_words else 0 for word in user_input.split()]
    padded_sequence = sequence.pad_sequences([input_sequence], maxlen=max_len)
    return padded_sequence



#building the main app
st.title('App for Classification Task')

option = st.selectbox(
    'Select a classification task',
    ('Select', 'Sentiment Classification', 'Image Classification')
)

if option == 'Select':
    pass

elif option == 'Sentiment Classification':
    st.header('Sentiment Classification')
    st.subheader('Select your model')

    sentiment_model = st.radio(
        'Select Model',
        ('RNN', 'BackPropagation', 'Perceptron', 'DNN', 'LSTM')
    )

    if sentiment_model == 'RNN':
        st.write(f'You selected {sentiment_model} model for sentiment classification')

        

        # Input field for user to enter text
        user_input = st.text_area("Enter text for prediction", "")
        if st.button("Predict"):
            if user_input:
                prediction_result = rnn_predict(user_input)
                st.write(f"The message is classified as: {prediction_result}")
            else:
                st.write("Please enter some text for prediction")
    elif sentiment_model == 'DNN':  # New block for DNN option
            st.header('DNN Movie Sentiment Classification')
            
            user_review = st.text_area("Enter your movie review here:", "")

            if st.button('Predict'):
                if user_review:
                    prediction_result = predict_dnnsentiment(user_review)
                    st.write(f"The review is classified as: {prediction_result}")
                else:
                    st.write("Please enter a movie review for prediction")
    elif sentiment_model== 'LSTM':
            st.header('LSTM review Classification')
            st.subheader('review classification using LSTM model')
            user_review = st.text_area("Enter your movie review here:", "")
            if st.button('Predict'):
                if user_review:
                    prediction_result = predict_sentiment(user_review)
                    st.write(f"The review is classified as: {prediction_result}")
                else:
                    st.write("Please enter a movie review for prediction")
    elif sentiment_model=='BackPropagation':
            st.header('Movie Review Classification ')
            st.subheader('Movie Review Classification using Backpropagation model')
            user_input = st.text_area("Enter your review ")

            if st.button('Predict'):
                words = user_input.split()
                word_index = imdb.get_word_index()
                review = [word_index[word] if word in word_index and word_index[word] < 10000 else 0 for word in words]
                review = pad_sequences([review], maxlen=200)
                
                prediction = BP_model.predict(review)
                sentiment = "Positive" if prediction[0] == 1 else "Negative"
                
                st.write(f"Predicted Sentiment: {sentiment}")
    elif sentiment_model=='Perceptron':
            st.header('Movie Review Classification ')
            st.subheader('Movie Review Classification using Perceptron model')
            user_input = st.text_area("Enter your review ")

            if st.button('Predict'):
                processed_input = preprocess_input(user_input)
                prediction = pm_model.predict(processed_input)[0]
                sentiment = "Positive" if prediction == 1 else "Negative"
                st.write(f"Predicted Sentiment: {sentiment}")

elif option == 'Image Classification':  # Updated for Image Classification option
    st.header('Image Classification')
    st.subheader('Your default model is CNN')  # Added subheader for default CNN model
    imagetask = st.radio(
        'Select task',
        ('Tumour detection','Digit Recognition')
    )
    if imagetask=='Tumour detection':
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

            if st.button('Predict'):
                prediction_result = make(uploaded_file)  # Use the function for CNN prediction
                st.write(f"The image is classified as: {prediction_result}")
    elif imagetask=='Digit Recognition':
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","webp"])
        if uploaded_file:
                st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
                if st.button('Predict'):
                    prediction_result = predict_digit(uploaded_file)  # Use the function for CNN prediction
                    st.write(f"The image is classified as: {prediction_result}")
