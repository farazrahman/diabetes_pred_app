# Use the official Python image as the base image
FROM python:3.11

# Set the working directory in the container
WORKDIR /diabetes_pred_app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit application files into the container
COPY . .

# Expose the port that Streamlit will be running on
EXPOSE 8501

# Set the default command to run when the container starts
CMD ["streamlit", "run", "--server.enableCORS", "false", "app.py"]