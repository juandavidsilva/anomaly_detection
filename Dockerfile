FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x ./main.py

ENTRYPOINT ["python", "./main.py"]

CMD ["/input/demo.json", "output"]

