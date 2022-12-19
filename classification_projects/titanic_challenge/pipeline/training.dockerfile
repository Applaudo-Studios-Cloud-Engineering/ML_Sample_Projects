FROM python:3.10-alpine

RUN apk add --update --no-cache py3-numpy py3-pandas py3-scikit-learn
ENV PYTHONPATH=/usr/lib/python3.10/site-packages

WORKDIR /work
COPY src/*.py ./src/
# RUN mkdir data
# COPY data ./data
RUN mkdir data
RUN mkdir models

WORKDIR /work/src

# ENTRYPOINT ["python", "main.py"]