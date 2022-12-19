FROM python:3.10-alpine

RUN pip install --no-cache-dir --upgrade fastapi uvicorn pydantic
RUN apk add --update --no-cache py3-scikit-learn
ENV PYTHONPATH=/usr/lib/python3.10/site-packages

WORKDIR /work
COPY src/*.py ./src/
COPY model ./model

WORKDIR /work/src

ENTRYPOINT ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "80"]