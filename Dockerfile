FROM python:3.11-slim-buster

WORKDIR /app

COPY requirements.txt ./

RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt


COPY . .

EXPOSE 5000

CMD [ "flask", "run", "--host", "0.0.0.0" ]
