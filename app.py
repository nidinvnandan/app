from main import rnn_predict
import streamlit as st
from PIL import Image
import numpy as np
from numpy import argmax
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence 
import pickle
from sklearn.datasets import load_iris
try:
    modelcnn1 = tf.keras.models.load_model('cnnmodel.h5')
except Execution as e:
    st.error('f"error loading: {e}")
             
irismodel=tf.keras.models.load_model('iris_model.h5')
lstmmodel = tf.keras.models.load_model('lstm_imdb_model.h5')
digitmodel=tf.keras.models.load_model('digit.h5')
iris=load_iris()
class_names = iris.target_names

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
    

def predict_iris_with_saved_model(user_input):
    # Load the model (replace irismodel with your loaded model)
    

    # Convert user input to float values
    user_input = [float(val) for val in user_input]

    # Predict using the loaded model
    prediction = irismodel.predict(np.array([user_input]))
    predicted_class = np.argmax(prediction)

    name = class_names[predicted_class]

    st.write(f'Predicted class: {name}')

            
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
    # Convert the image to grayscale if necessary
    img = img.convert('L')  # Convert to grayscale

    # Resize the image to match MNIST dataset's input shape (28x28 pixels)
    img = img.resize((28, 28))

    # Convert image to array and normalize
    img_array = np.array(img)
    img_array = img_array.reshape((1, 28, 28, 1))  # Reshape to match model input shape
    img_array = img_array.astype('float32') / 255.0  # Normalize pixels to range [0, 1]

    return img_array

def predict_digit(image):
    # Preprocess the image (reshape if necessary, normalize, etc.)
    # Make predictions using the loaded model
    image = preprocess_user_image(image)  # Replace preprocess_image with your actual preprocessing steps
    prediction = digitmodel.predict(image)
    predicted_label = argmax(prediction, axis=1)
    return predicted_label[0]

with open('backprop_model.pkl', 'rb') as file:
    BP_model = pickle.load(file)
with open('perceptron_model.pkl','rb') as file:
     p_model=pickle.load(file)
st.title('App')

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
            st.header('DNN Iris Classification')
            st.subheader('Iris classification using DNN model')
            # Input fields for separate numerical values
            sepal_length = st.number_input("Enter Sepal Length:")
            sepal_width = st.number_input("Enter Sepal Width:")
            petal_length = st.number_input("Enter Petal Length:")
            petal_width = st.number_input("Enter Petal Width:")

            if st.button('Predict'):
                user_input = [sepal_length, sepal_width, petal_length, petal_width]
                predict_iris_with_saved_model(user_input)
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
            st.header('Iris Classification ')
            st.subheader('Iris classification using Backpropagation model')
            sepal_length = st.number_input("Enter Sepal Length:")
            sepal_width = st.number_input("Enter Sepal Width:")
            if st.button('Predict'):
                user_input = np.array([[sepal_length, sepal_width]])
                user_prediction = BP_model.predict(user_input)[0]
                predicted_class_name = class_names[user_prediction]
                st.write(f"Predicted class for user input: {predicted_class_name}")
    elif sentiment_model=='Perceptron':
            st.header('Iris Classification ')
            st.subheader('Iris classification using Perceptron model')
            sepal_length = st.number_input("Enter Sepal Length:")
            sepal_width = st.number_input("Enter Sepal Width:")
            if st.button('Predict'):
                user_input = np.array([[sepal_length, sepal_width]])
                user_prediction = p_model.predict(user_input)[0]
                predicted_class_name = class_names[user_prediction]
                st.write(f"Predicted class for user input: {predicted_class_name}")

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
