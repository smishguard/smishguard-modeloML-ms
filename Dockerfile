# Usa una imagen base oficial de Python
FROM python:3.9

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar los archivos necesarios al contenedor
COPY app.py /app
COPY spam_classifier_model.keras /app

# Instalar las dependencias
RUN pip install flask tensorflow nltk

# Descargar los datos de NLTK necesarios
RUN python -c "import nltk; nltk.download('stopwords')"

# Exponer el puerto que usará Flask
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
