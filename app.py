import logging
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from deep_translator import GoogleTranslator

nltk.download('stopwords')

app = Flask(__name__)

# Configuración del log
logging.basicConfig(level=logging.INFO)

# Cargar el modelo
try:
    model = tf.keras.models.load_model('spam_classifier_model.keras')
    logging.info("Modelo cargado correctamente")
except Exception as e:
    logging.error(f"Error al cargar el modelo: {e}")
    model = None

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
    try:
        # Verificar si el modelo se cargó correctamente
        if model is None:
            return jsonify({'error': 'El modelo no está disponible'}), 500

        # Obtener el texto del POST
        data = request.get_json(force=True)
        text = data.get('text', '')

        # Validar que el texto no esté vacío
        if not text:
            return jsonify({'error': 'El texto proporcionado está vacío'}), 400

        # Traducir el texto al inglés
        try:
            translated_text = GoogleTranslator(source='auto', target='en').translate(text)
            logging.info(f"Texto original: {text}")
            logging.info(f"Texto traducido: {translated_text}")
        except Exception as e:
            logging.error(f"Error al traducir el texto: {e}")
            return jsonify({'error': 'Error al traducir el texto', 'details': str(e)}), 500

        # Preprocesar el texto traducido
        processed_text = preprocess_message(translated_text)

        # Realizar la predicción
        prediction = model.predict(processed_text)

        # Determinar si es spam o no
        logging.info(f"Predicción bruta: {prediction}")
        result = "spam" if prediction > 0.05 else "not spam"
        
        return jsonify({'prediction': result})
    except Exception as e:
        logging.error(f"Error durante la predicción: {e}")
        return jsonify({'error': 'Error en el servidor', 'details': str(e)}), 500

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Spam Detection Service!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
