# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    git \ 
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    

# Copy the rest of the application code into the container
COPY . .

# Ensure the entrypoint script is executable
RUN chmod +x /app/scripts/startup.sh

# Set the entrypoint to run the script
CMD ["/app/scripts/startup.sh"]