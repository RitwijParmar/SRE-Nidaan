# ── Dockerfile for The Body (FastAPI Backend) ──
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements-serving.txt ./requirements-serving.txt
RUN pip install --no-cache-dir -r requirements-serving.txt

# Copy application code
COPY backend/ ./backend/
COPY src/ ./src/
COPY ops/ ./ops/
COPY config.py .
RUN mkdir -p /app/feedback

ENV PYTHONPATH=/app

# Expose FastAPI port
EXPOSE 8001

# Run the server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8001"]
