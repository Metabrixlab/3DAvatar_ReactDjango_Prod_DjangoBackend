FROM ubuntu:20.04
RUN echo "Ubuntu Had setup"

# Disable timezone prompt
ENV DEBIAN_FRONTEND=noninteractive
# Install.
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip

RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY . /code/

# Expose the port on which your Django app will run (change if necessary)
EXPOSE 8000

# Run the Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]