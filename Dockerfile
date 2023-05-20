FROM python:3.9
WORKDIR /code/bow
ADD . .
RUN python -m pip install --upgrade pip \
	&& pip install -r requirements.txt
CMD ["python", "./test.py"]
