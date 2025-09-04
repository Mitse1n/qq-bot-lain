# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Force python to print directly to stdout
ENV PYTHONUNBUFFERED=1

# Copy the project's dependency configuration
COPY pyproject.toml ./

# Install any needed packages specified in pyproject.toml
# Install the project in editable mode with its dependencies
RUN pip install .

# Copy the rest of the application's source code from the current directory to the working directory in the container
COPY . .

# Run main.py when the container launches
CMD ["python3", "main.py"]
