FROM python:3.8-slim-buster

VOLUME /Dashboard

# Create a working directory.
RUN mkdir wd
WORKDIR wd

# Install Python dependencies.
COPY requirements.txt .
RUN pip3 install -r requirements.txt
EXPOSE 8081

# Copy the rest of the codebase into the image
WORKDIR Dashboard
COPY Dashboard/ .

# Finally, run gunicorn.
CMD [ "gunicorn", "--workers=5", "--threads=1", "-b 0.0.0.0:8081", "app:server"]
