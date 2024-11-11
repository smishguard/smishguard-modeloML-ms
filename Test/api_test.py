import pytest
import requests
import json

# Define la URL de la API
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
