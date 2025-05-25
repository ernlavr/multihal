FROM python:3.11-slim

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY config/ /app/config
COPY requirements.txt /app
COPY src/ /app/src
COPY main.py /app
COPY res/ /app/res
COPY .env /app


RUN pip install -r /app/requirements.txt --no-cache-dir
# RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

ENTRYPOINT ["python", "-u", "main.py", "--config", "config/default.yaml"]