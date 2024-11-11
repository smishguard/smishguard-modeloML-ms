import pandas as pd
import requests
from tqdm import tqdm

def load_data(file_path):
    # Cargar solo las primeras 200 filas del archivo CSV
    df = pd.read_csv(file_path, encoding='ISO-8859-1', nrows=1000)
    # Seleccionar las columnas 'LABEL' y 'TEXT' para la clase y el mensaje
    df = df[['LABEL', 'TEXT']]
    df.columns = ['class', 'text']  # Renombrar las columnas para simplificar
    return df

def send_to_service(text):
    url = "http://127.0.0.1:5000/predict"  # Cambia la URL si es necesario
    payload = {"text": text}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("prediction")
    except requests.RequestException as e:
        print(f"Error en la solicitud: {e}")
        return "error"

def evaluate_model(data):
    correct_predictions = 0
    total_messages = len(data)  # Total de mensajes ahora será 200

    for _, row in tqdm(data.iterrows(), total=total_messages, desc="Procesando mensajes"):
        # Convertir 'ham' a 'not spam' y 'spam' o 'Smishing' a 'spam' para coincidir con la respuesta del servicio
        actual_class = "spam" if row['class'].lower() in ["spam", "smishing"] else "not spam"
        predicted_class = send_to_service(row['text'])
        
        if predicted_class == actual_class:
            correct_predictions += 1

    accuracy = (correct_predictions / total_messages) * 100
    print(f"\nPrecisión del modelo: {accuracy:.2f}%")

# Ruta del archivo CSV
file_path = "C:/Users/ivana/Desktop/Microservicios Tesis/smishguard-modeloML-ms/Dataset_5971.csv"  # Cambia la ruta según sea necesario
data = load_data(file_path)
evaluate_model(data)
