FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app
COPY . /app
EXPOSE 8000
CMD ["uvicorn","localserve:app","--host","0.0.0.0"]
