from locust import HttpUser, task, between

class SpamDetectionUser(HttpUser):
    wait_time = between(1, 3)  # Tiempo de espera entre cada tarea

    @task
    def predict_spam(self):
        # Simular un mensaje de prueba
        payload = {
            "text": "Este es un mensaje de prueba para verificar el servicio"
        }
        self.client.post("/predict", json=payload)
