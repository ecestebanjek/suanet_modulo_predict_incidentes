# suanet_modulo_predict_incidentes
Este es el código SDM de SUANET para el modulo de predicción de incidentes. Este es el modulo 3.

# Pasos para crear la imagen de docker
    docker build -t jcastrosdm1/img_mod3:1.0 .
    docker login
    docker push jcastrosdm1/img_mod3:1.0

# Pasos para desplegar en servidor
    docker pull jcastrosdm1/img_mod3:1.0
    docker create p8503:8501 --name cont_mod1 jcastrosdm1/img_mod3:1.0
    docker start cont_mod3 -d

Nota: El puerto dentro del contenedor siempre sera 8501 para streamlit, y el puerto del servidor (el primero) puede incrementar o variar.

