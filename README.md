# SmishGuard - Spam Detection Microservice

## Descripción

Este microservicio ofrece una API para la detección de spam en mensajes de texto. Utiliza un modelo de machine learning entrenado para clasificar mensajes como `spam` o `not spam`. El servicio está diseñado para integrarse en el ecosistema de `SmishGuard`, ayudando a identificar posibles mensajes de phishing y spam en aplicaciones de mensajería.

El modelo está entrenado en TensorFlow y se implementa en Flask para servir predicciones a través de un endpoint HTTP.

## Endpoints

### 1. `GET /`

#### Descripción
Endpoint básico para verificar que el servicio esté funcionando.

#### Respuesta
```json
{
    "message": "Welcome to the Spam Detection Service!"
}
```

### 2. `POST /predict`

#### Descripción
Recibe un mensaje de texto y devuelve una predicción de si el mensaje es spam o not spam.

#### Solicitud
- **URL:** `/predict`
- **Método:** `POST`
- **Body:** JSON con el campo `text`.

```json
{
    "text": "Texto del mensaje a analizar"
}
```

#### Respuesta
Un JSON que contiene la predicción de la clase del mensaje (`spam` o `not spam`).

```json
{
    "prediction": "spam"
}
```

#### Errores
Si el texto no está incluido o el modelo no está disponible, se devuelve un error con un mensaje adecuado:

```json
{
    "error": "El texto proporcionado está vacío"
}
```

## Instalación y Ejecución Local

1. Clonar el repositorio
    ```bash
    git clone <URL_del_repositorio>
    cd smishguard-modeloML-ms
    ```

2. Instalar las dependencias necesarias
    Se recomienda usar un entorno virtual.
    ```bash
    python -m venv venv
    source venv/bin/activate  # Para Windows, usa `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. Descargar y colocar el modelo entrenado
    Coloca el archivo del modelo entrenado `spam_classifier_model.keras` en el directorio raíz del proyecto.

4. Ejecutar la aplicación
    ```bash
    python app.py
    ```
    El servicio estará disponible en `http://127.0.0.1:5000`.

## Pruebas

### Pruebas Unitarias
Se realizaron pruebas unitarias para verificar el funcionamiento de los endpoints y la lógica del modelo.

![Resultado de Pruebas Unitarias](Test/Resultado_pruebas_unitarias.jpg)

### Pruebas de la API
Se llevaron a cabo pruebas de la API para confirmar que los endpoints responden correctamente y en el formato esperado.
![Resultado de Pruebas API](Test/Resultado_pruebas_api_ML.jpg)

### Pruebas de Carga
Se realizaron pruebas de carga utilizando Locust para simular múltiples usuarios concurrentes accediendo al endpoint `/predict`. Los resultados mostraron que el servicio puede manejar la carga esperada de manera eficiente.

## Evaluación del Modelo
Se utilizó un conjunto de datos para evaluar la precisión del modelo. El script de evaluación carga los datos y envía cada mensaje al servicio para obtener la predicción, calculando luego la precisión general.

```python
def evaluate_model(data):
    correct_predictions = 0
    total_messages = len(data)

    for _, row in data.iterrows():
        actual_class = "spam" if row['class'].lower() in ["spam", "smishing"] else "not spam"
        predicted_class = send_to_service(row['text'])
        
        if predicted_class == actual_class:
            correct_predictions += 1

    accuracy = (correct_predictions / total_messages) * 100
    print(f"\nPrecisión del modelo: {accuracy:.2f}%")
```

### Resultado de la evaluación
El modelo logró una precisión del X% en la detección de mensajes de spam y no spam.

## Contribuciones
Se aceptan contribuciones al proyecto. Por favor, abre un issue para discutir los cambios que te gustaría realizar.

## Licencia
Este proyecto está licenciado bajo la MIT License.