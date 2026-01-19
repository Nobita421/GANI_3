FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose ports for Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

CMD ["bash"]
