FROM python:3.12-slim

WORKDIR /app

# Install CPU-only torch + transformers + server deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
