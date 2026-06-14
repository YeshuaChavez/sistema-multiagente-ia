# Usar una imagen oficial de Python slim como base
FROM python:3.11-slim

# Evitar que Python escriba archivos .pyc y habilitar modo unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias de sistema mínimas si fueran necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de dependencias e instalarlas
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copiar directorios necesarios
COPY backend /app/backend
COPY agentes /app/agentes
COPY Base?de?Datos /app/Base_de_Datos
RUN ln -s /app/Base_de_Datos "/app/Base de Datos"

# Exponer el puerto
EXPOSE 8080

# Comando para iniciar la aplicación mediante Uvicorn usando el puerto dinámico de Heroku
CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT
