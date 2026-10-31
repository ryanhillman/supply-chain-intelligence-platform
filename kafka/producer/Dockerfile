FROM python:3.11-slim

WORKDIR /app

# Build context is the repository root (see kafka/docker-compose.yml), so the
# world model in simulation/ is copied next to the producer script.
COPY kafka/producer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY simulation/ simulation/
COPY kafka/producer/simulate.py .

CMD ["python", "-u", "simulate.py"]
