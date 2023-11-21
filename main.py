import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#RNN



dataset = pd.read_csv('C:/Users/Admin/Documents/SMSSpamCollection',sep='\t',names=['label','message'])

dataset['label'] = dataset['label'].map( {'spam': 1, 'ham': 0} )
X = dataset['message'].values
y = dataset['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
tokeniser = tf.keras.preprocessing.text.Tokenizer()
tokeniser.fit_on_texts(X_train)
encoded_train = tokeniser.texts_to_sequences(X_train)
encoded_test = tokeniser.texts_to_sequences(X_test)

max_length = 10
padded_train = tf.keras.preprocessing.sequence.pad_sequences(encoded_train, maxlen=max_length, padding='post')
padded_test = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')


vocab_size = len(tokeniser.word_index)+1

# define the model




model=tf.keras.models.Sequential([
   tf.keras.layers.Embedding(input_dim=vocab_size,output_dim= 24, input_length=max_length),
   tf.keras.layers.SimpleRNN(24, return_sequences=False),
   tf.keras.layers.Dense(64, activation='relu'),
   tf.keras.layers.Dense(32, activation='relu'),
   tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='min', patience=10)


# fit the model
model.fit(x=padded_train,
         y=y_train,
         epochs=50,
         validation_data=(padded_test, y_test),
         callbacks=[early_stop]
         )
print("----------------------  -------------------------\n")






model.save("model_kerasrnn.h5")
modelrnn = tf.keras.models.load_model('model_kerasrnn.h5')

def rnn_predict(input_text):
    # Process input text similarly to training data
    encoded_input = tokeniser.texts_to_sequences([input_text])
    padded_input = tf.keras.preprocessing.sequence.pad_sequences(encoded_input, maxlen=max_length, padding='post')
    prediction = modelrnn.predict(padded_input)
    if prediction > 0.5:
        return "spam"
    else:
        return "ham"
    






