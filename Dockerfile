# app/Dockerfile

FROM python:3.11

WORKDIR /mod3_planea_incidentes

RUN pip3 install --upgrade pip

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ecestebanjek/suanet_modulo_predict_incidentes.git .

RUN pip3 install -r requirements.txt
RUN pip3 install tensorflow==2.14.0

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app_pred_inc.py", "--server.port=8501", "--server.address=0.0.0.0"]