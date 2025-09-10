# Use an official Python runtime as a parent image
FROM python:3.13-slim

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

# Copy and set executable permissions for the start script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Run start.sh when the container launches
CMD ["/app/start.sh"]
