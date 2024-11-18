# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim


# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Set working environment


ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80
EXPOSE 5000
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app

ENTRYPOINT ["flask", "run"]