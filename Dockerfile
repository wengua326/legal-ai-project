# Step 1
FROM python:3.11-slim

# Step 2
WORKDIR /app

# Step 3
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

# Step 4
COPY . .

# Step 5
RUN pip install --no-cache-dir -r requirements.txt

# Step 6
RUN unzip -o chroma_db.zip && unzip -o docstore.zip && rm *.zip

# Step 7
EXPOSE 7860

# Step 8 : run api.py
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
