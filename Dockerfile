FROM python:3.8-slim-buster

# Create a working directory.
RUN mkdir wd
WORKDIR wd

# Install Python dependencies.
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy the rest of the codebase into the image
VOLUME /Dashboard
COPY . ./Dashboard

EXPOSE 8081

# Finally, run gunicorn.
CMD [ "gunicorn", "--workers=5", "--threads=1", "-b 0.0.0.0:8081", "app:server"]
