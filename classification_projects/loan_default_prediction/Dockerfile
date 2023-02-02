FROM python:3.10

WORKDIR /code

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r ./requirements.txt

COPY ./src ./src
COPY src/main.py ./

CMD ["python", "main.py"]