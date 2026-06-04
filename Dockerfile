FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY src/ src/
COPY app/ app/
COPY api/ api/
COPY data/processed/cleaned_data.csv data/processed/cleaned_data.csv
COPY models/ models/

EXPOSE 8501 8000

CMD ["streamlit", "run", "app/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
