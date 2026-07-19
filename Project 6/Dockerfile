FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
# model_artifacts.joblib is copied in for local/default mode; when MODEL_SOURCE=blob
# is set at deploy time, the app pulls the model from Azure Blob Storage instead.
COPY model_artifacts.joblib .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
