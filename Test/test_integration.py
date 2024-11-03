# test_integration.py

import pytest
import json
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True

    with app.test_client() as client:
        yield client

# Prueba para verificar que el servicio está activo
def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.json.get("message") == "Welcome to the Spam Detection Service!"

# Prueba para verificar predicción con texto de spam
def test_predict_spam(client):
    payload = {"text": "¡Dinero gratis ahora mismo!"}
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    assert response.json.get("prediction") in ["spam", "not spam"]

# Prueba para verificar predicción con texto no spam
def test_predict_not_spam(client):
    payload = {"text": "Hola, solo quería verificar el estado del proyecto."}
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    assert response.json.get("prediction") in ["spam", "not spam"]

# Prueba de error cuando el texto está vacío
def test_predict_empty_text(client):
    payload = {"text": ""}
    response = client.post('/predict', json=payload)
    assert response.status_code == 400
    assert response.json.get("error") == "El texto proporcionado está vacío"

# Prueba cuando el modelo no está disponible
def test_predict_no_model(client, monkeypatch):
    # Simular que el modelo es None
    def mock_load_model(*args, **kwargs):
        return None
    monkeypatch.setattr("tensorflow.keras.models.load_model", mock_load_model)

    # Necesitamos reinicializar la aplicación para que cargue el modelo simulado
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        payload = {"text": "Texto de prueba"}
        response = client.post('/predict', json=payload)
        assert response.status_code == 500
        assert response.json.get("error") == "El modelo no está disponible"

# Prueba de error en la traducción
def test_predict_translation_error(client, monkeypatch):
    # Simular un error en la traducción
    def mock_translate(*args, **kwargs):
        raise Exception("Error de traducción")
    monkeypatch.setattr("deep_translator.GoogleTranslator.translate", mock_translate)

    payload = {"text": "Texto de prueba"}
    response = client.post('/predict', json=payload)
    assert response.status_code == 500
    assert "Error al traducir el texto" in response.json.get("error")
