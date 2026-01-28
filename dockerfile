FROM python:3.10-slim

WORKDIR /app

COPY flask_app/ /app/

COPY models/vectorizer.pkl /app/models/vectorizer.pkl

# Upgrade pip first (fixes many download issues)
RUN pip install --upgrade pip

# Install requirements with a 1000-second timeout (default is usually 15s)
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt 

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

#local
# CMD ["python", "app.py"]  

#Prod
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]