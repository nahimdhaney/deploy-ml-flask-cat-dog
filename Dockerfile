FROM python:3.6-slim
COPY ./flaskApp/predictApp.py /deploy/
COPY ./flaskApp/trained_model.pkl /deploy/
COPY ./requirements.txt /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 5000
COPY ./flaskApp/templates/home.html /deploy/templates/
ENTRYPOINT ["python", "predictApp.py"]