from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

app = Flask(__name__)

# Cargar el modelo
try:
    model = tf.keras.models.load_model('spam_classifier_model.keras')
    print("Modelo cargado correctamente")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

model = tf.keras.models.load_model('spam_classifier_model.keras')

# Definir parámetros utilizados en el preprocesamiento
vocab_size = 10000
sentence_len = 200
stemmer = PorterStemmer()

def preprocess_message(message):
    # Limpiar el mensaje
    message = re.sub("[^a-zA-Z]", " ", message)
    message = message.lower()
    message = message.split()
    message = [stemmer.stem(word) for word in message if word not in set(stopwords.words("english"))]
    message = " ".join(message)
    
    # Codificación one-hot y padding
    one_hot_encoded = one_hot(message, vocab_size)
    text = pad_sequences([one_hot_encoded], maxlen=sentence_len, padding="pre")
    return text

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener el texto del POST
    data = request.get_json(force=True)
    text = data['text']

    # Preprocesar el texto
    processed_text = preprocess_message(text)

    # Realizar la predicción
    prediction = model.predict(processed_text)

    # Determinar si es spam o no
    print(f"Predicción bruta: {prediction}")
    result = "spam" if prediction > 0.01 else "not spam"
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

