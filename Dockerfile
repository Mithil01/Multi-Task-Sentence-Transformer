
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app/

# Expose port for Streamlit
EXPOSE 8502

# Command to run
CMD ["streamlit", "run", "src/app.py", "--server.port=8502", "--server.address=0.0.0.0"]