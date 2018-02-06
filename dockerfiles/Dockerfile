# add 7z tar and zip archivers
FROM nvidia/cuda:8.0-cudnn6-devel

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:Ubuntu@41' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN apt -y install libgl1-mesa-glx

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

# writing env variables to /etc/profile as mentioned here https://docs.docker.com/engine/examples/running_ssh_service/#run-a-test_sshd-container
RUN echo "export CONDA_DIR=/opt/conda" >> /etc/profile
RUN echo "export PATH=$CONDA_DIR/bin:$PATH" >> /etc/profile

RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    apt-get update && \
    apt-get install -y wget git libhdf5-dev g++ graphviz openmpi-bin nano && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    ln /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/local/cuda/lib64/libcudnn.so && \
    ln /usr/lib/x86_64-linux-gnu/libcudnn.so.6 /usr/local/cuda/lib64/libcudnn.so.6 && \
    ln /usr/include/cudnn.h /usr/local/cuda/include/cudnn.h  && \
    rm Miniconda3-4.2.12-Linux-x86_64.sh

ENV NB_USER keras
ENV NB_UID 1000

RUN echo "export NB_USER=keras" >> /etc/profile
RUN echo "export NB_UID=1000" >> /etc/profile

RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" >> /etc/profile
RUN echo "export CPATH=/usr/include:/usr/include/x86_64-linux-gnu:/usr/local/cuda/include:$CPATH" >> /etc/profile
RUN echo "export LIBRARY_PATH=/usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LIBRARY_PATH" >> /etc/profile
RUN echo "export CUDA_HOME=/usr/local/cuda" >> /etc/profile
RUN echo "export CPLUS_INCLUDE_PATH=$CPATH" >> /etc/profile
RUN echo "export KERAS_BACKEND=tensorflow" >> /etc/profile

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \ 
    chown keras $CONDA_DIR -R  

USER keras

RUN  mkdir -p /home/keras/notebook

# Python
ARG python_version=3.5

RUN conda install -y python=${python_version} && \
    pip install --upgrade pip && \
    pip install tensorflow-gpu==1.4 && \
    conda install Pillow scikit-learn notebook pandas matplotlib mkl nose pyyaml six h5py && \
    conda install theano pygpu bcolz && \
    pip install keras kaggle-cli lxml opencv-python requests scipy tqdm visdom && \
    conda install pytorch torchvision cuda80 -c soumith && \
    pip install imgaug && \
    conda clean -yt

RUN pip install git+https://github.com/ipython-contrib/jupyter_contrib_nbextensions && \
    jupyter contrib nbextension install --user

RUN conda install -y -c conda-forge cython
RUN conda install -y -c conda-forge libgdal
RUN conda install -y -c conda-forge gdal
RUN conda install -y -c conda-forge scikit-image
RUN conda install -y -c conda-forge pyproj
RUN conda install -y -c conda-forge geopandas
RUN conda install -y -c conda-forge tqdm
RUN conda install -y -c conda-forge shapely=1.5.16
RUN conda install -y -c conda-forge scipy
RUN conda install -y -c conda-forge networkx=1.11
RUN conda install -y -c conda-forge fiona
RUN pip install utm
RUN pip install osmnx==0.5.1
RUN pip install numba
RUN conda install -y -c conda-forge scikit-image

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV CPATH /usr/include:/usr/include/x86_64-linux-gnu:/usr/local/cuda/include:$CPATH
ENV LIBRARY_PATH /usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LIBRARY_PATH
ENV CUDA_HOME /usr/local/cuda
ENV CPLUS_INCLUDE_PATH $CPATH
ENV KERAS_BACKEND tensorflow

WORKDIR /home/keras/notebook

EXPOSE 8888 6006 22 8097

CMD jupyter notebook --port=8888 --ip=0.0.0.0 --no-browser
