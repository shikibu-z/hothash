FROM python:3.9
WORKDIR /app
RUN python3 -m pip install -U flask jsonlines google-cloud google-cloud-storage google-api-python-client gunicorn numpy pyheaven tqdm
COPY . .
CMD ["python3", "worker.py"]
