<<<<<<< HEAD
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
=======
FROM python:3.9

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirement.txt

CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]
>>>>>>> 8c9334a (First commit)
