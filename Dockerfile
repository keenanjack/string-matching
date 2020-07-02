FROM python:3
ENV PYTHONUNBUFFERED 1
WORKDIR /app
COPY . /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 80
CMD ["python","app.py"]