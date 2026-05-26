FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY data/ /app/data/

EXPOSE 8000

# 6. Comando para arrancar la API dentro de Docker (quitamos el --reload para producción)
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]