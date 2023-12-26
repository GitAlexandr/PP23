# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Set working directory
WORKDIR /actions

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Run the application
CMD ["uvicorn", "actions/main:app", "--host", "0.0.0.0", "--port", "8000"]
