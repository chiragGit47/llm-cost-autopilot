# ==========================================================
# Base Python image
# ==========================================================

FROM python:3.11.9-slim


# ==========================================================
# Python environment settings
# ==========================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# ==========================================================
# Application directory
# ==========================================================

WORKDIR /app


# ==========================================================
# Install Python dependencies first
#
# We copy requirements separately so Docker can cache this
# layer when application code changes.
# ==========================================================

COPY requirements.txt .


RUN pip install \
    --no-cache-dir \
    -r requirements.txt


# ==========================================================
# Copy application
# ==========================================================

COPY . .


# ==========================================================
# FastAPI port
# ==========================================================

EXPOSE 8000


# ==========================================================
# Start application
# ==========================================================

CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]