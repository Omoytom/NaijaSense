# 1. Use a lightweight, official Python runtime base
FROM python:3.11-slim

# 2. Set system environment variables to optimize Python inside a container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Establish the operational directory inside the container
WORKDIR /app

# 4. Install system dependencies required for data processing libraries if any
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy over your dependency map first (takes advantage of Docker caching layers)
COPY requirements.txt .

# 6. Install the Python package registry
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the entire codebase into the container execution context
COPY . .

# 8. Expose the port FastAPI runs on
EXPOSE 8000

# 9. Execute Uvicorn to host the app live
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]