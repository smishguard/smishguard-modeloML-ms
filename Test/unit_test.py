import unittest
from flask import json
from app import app  # Importar el servicio Flask desde el archivo app.py
import tensorflow as tf

class SpamDetectionServiceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Configurar el cliente de prueba para el servicio Flask
        cls.client = app.test_client()
        cls.client.testing = True

    def test_home_route(self):
        """Prueba de acceso a la ruta principal"""
        response = self.client.get('/')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('message', data)
        self.assertEqual(data['message'], "Welcome to the Spam Detection Service!")

    def test_predict_spam_message(self):
        """Prueba de predicción: Mensaje spam"""
        response = self.client.post(
            '/predict',
            json={'text': "Congratulations! You won a free prize. Call now!"}
        )
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', data)
        self.assertEqual(data['prediction'], "spam")

    def test_predict_ham_message(self):
        """Prueba de predicción: Mensaje no spam (ham)"""
        response = self.client.post(
            '/predict',
            json={'text': "Hey, just wanted to say hello!"}
        )
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', data)
        self.assertEqual(data['prediction'], "not spam")

    def test_empty_text_error(self):
        """Prueba de error: texto vacío"""
        response = self.client.post('/predict', json={'text': ""})
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)
        self.assertEqual(data['error'], "El texto proporcionado está vacío")

    def test_missing_text_key_error(self):
        """Prueba de error: falta la clave 'text' en la solicitud"""
        response = self.client.post('/predict', json={})
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)
        self.assertEqual(data['error'], "El texto proporcionado está vacío")

    def test_model_not_loaded_error(self):
        """Prueba de error: modelo no disponible"""
        # Simular que el modelo no se ha cargado, se establece en None
        app.config['model'] = None  # Modificar temporalmente el modelo a None
        
        response = self.client.post(
            '/predict',
            json={'text': "Test message"}
        )
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 500)
        self.assertIn('error', data)
        self.assertEqual(data['error'], "El modelo no está disponible")

        # Restaurar el modelo para evitar problemas en otras pruebas
        app.config['model'] = tf.keras.models.load_model('/app/spam_classifier_model.keras')

    def test_translation_error_handling(self):
        """Prueba de manejo de errores en la traducción"""
        # Simular un error en la traducción
        with unittest.mock.patch('deep_translator.GoogleTranslator.translate', side_effect=Exception("Translation error")):
            response = self.client.post(
                '/predict',
                json={'text': "Mensaje de prueba"}
            )
            data = json.loads(response.data)
            
            self.assertEqual(response.status_code, 500)
            self.assertIn('error', data)
            self.assertEqual(data['error'], "Error al traducir el texto")

    def test_invalid_json_format(self):
        """Prueba de error: formato JSON no válido"""
        response = self.client.post(
            '/predict',
            data="Invalid JSON format",
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)
        self.assertIn('No se pudo procesar la solicitud JSON', data['error'])

if __name__ == "__main__":
    unittest.main()
