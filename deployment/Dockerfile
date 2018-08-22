FROM continuumio/miniconda3:4.5.4

EXPOSE 5000
WORKDIR /app
ENTRYPOINT ["python", "/app/app.py"]

COPY prod_requirements.txt .
RUN conda install --yes --file prod_requirements.txt

COPY . .

