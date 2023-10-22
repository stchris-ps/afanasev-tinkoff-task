FROM python:3.11

WORKDIR /workdir

COPY requirements.txt /workdir/requirements.txt
COPY code/ /workdir/code/


RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /workdir/requirements.txt

# Run the application
CMD ["uvicorn", "code.main:app", "--host", "0.0.0.0", "--port", "8080", --reload]
