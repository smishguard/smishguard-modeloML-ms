import pytest
import requests
import json

# Define la URL de la API. Si está en localhost, asegúrate de que la API esté en ejecución antes de las pruebas.
BASE_URL = "http://127.0.0.1:5000"

# Prueba para verificar que el servicio está activo
def test_home():
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    assert response.json().get("message") == "Welcome to the Spam Detection Service!"

# Prueba para verificar predicción con texto válido
def test_predict_spam():
    payload = {"text": "Free money now!!!"}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(f"{BASE_URL}/predict", data=json.dumps(payload), headers=headers)
    
    assert response.status_code == 200
    assert response.json().get("prediction") in ["spam", "not spam"]

# Prueba para verificar predicción con texto no spam
def test_predict_not_spam():
    payload = {"text": "Hello, I just wanted to check in on the project status."}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(f"{BASE_URL}/predict", data=json.dumps(payload), headers=headers)
    
    assert response.status_code == 200
    assert response.json().get("prediction") in ["spam", "not spam"]

# Prueba de error cuando el texto está vacío
def test_predict_empty_text():
    payload = {"text": ""}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(f"{BASE_URL}/predict", data=json.dumps(payload), headers=headers)
    
    assert response.status_code == 400
    assert response.json().get("error") == "El texto proporcionado está vacío"

# Prueba cuando el modelo no está disponible
def test_predict_no_model(monkeypatch):
    def mock_load_model(*args, **kwargs):
        return None

    monkeypatch.setattr("tensorflow.keras.models.load_model", mock_load_model)

    payload = {"text": "Test text"}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(f"{BASE_URL}/predict", data=json.dumps(payload), headers=headers)
    
    assert response.status_code == 500
    assert response.json().get("error") == "El modelo no está disponible"

# Prueba de error en la traducción
def test_predict_translation_error(monkeypatch):
    def mock_translate(*args, **kwargs):
        raise Exception("Translation error")

    monkeypatch.setattr("deep_translator.GoogleTranslator.translate", mock_translate)

    payload = {"text": "Test text"}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(f"{BASE_URL}/predict", data=json.dumps(payload), headers=headers)
    
    assert response.status_code == 500
    assert "Error al traducir el texto" in response.json().get("error")
