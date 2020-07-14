## About the project
###Major Components
1. Dockerfile
2. requirement.txt
3. Model File
4. Jenkins Jobs for automation.
 
#### 1. Dockerfile
First we'll create a Dockerfile to create a Docker Image  to setup environment with all dependencies installs.

Dockerfile looks like this.

```Dockerfile
FROM python:3.6

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
```

#### 1. requirements.txt
It has all liabraies listed for all environments (Keras, Sci-Kit Learn, Tensorflow)

Dependencies for Keras
```
    absl-py==0.9.0
    astunparse==1.6.3
    cachetools==4.1.1
    certifi==2020.6.20
    chardet==3.0.4
    gast==0.3.3
    google-auth==1.18.0
    google-auth-oauthlib==0.4.1
    google-pasta==0.2.0
    grpcio==1.30.0
    h5py==2.10.0
    idna==2.10
    importlib-metadata==1.7.0
    Keras==2.4.3
    Keras-Preprocessing==1.1.2
    Markdown==3.2.2
    numpy==1.19.0
    oauthlib==3.1.0
    opt-einsum==3.2.1
    protobuf==3.12.2
    pyasn1==0.4.8
    pyasn1-modules==0.2.8
    PyYAML==5.3.1
    requests==2.24.0
    requests-oauthlib==1.3.0
    rsa==4.6
    scipy==1.4.1
    six==1.15.0
    tensorboard==2.2.2
    tensorboard-plugin-wit==1.7.0
    tensorflow==2.2.0
    tensorflow-estimator==2.2.0
    termcolor==1.1.0
    urllib3==1.25.9
    Werkzeug==1.0.1
    wrapt==1.12.1
    zipp==3.1.0

```
Dependencies for Sci-Kit Learn
```
    joblib==0.16.0
    numpy==1.19.0
    scikit-learn==0.23.1
    scipy==1.5.1
    threadpoolctl==2.1.0
```

Dependencies for TensorFlow
```
    requests-oauthlib==1.3.0
    rsa==4.6
    scipy==1.4.1
    six==1.15.0
    tensorboard==2.2.2
    tensorboard-plugin-wit==1.7.0
    tensorflow==2.2.0
    tensorflow-estimator==2.2.0
    termcolor==1.1.0
    urllib3==1.25.9
    Werkzeug==1.0.1
    wrapt==1.12.1
    zipp==3.1.0
```
