FROM xapharius/baseminiconda3

COPY deployment /app

EXPOSE 5000

WORKDIR /app
ENTRYPOINT ["python", "/app/app.py"]

