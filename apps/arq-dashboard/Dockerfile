FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir arq-dashboard==0.1.4

COPY run.py /app/run.py

EXPOSE 8000

CMD ["python", "/app/run.py"]
