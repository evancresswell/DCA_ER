# Up to date Python version
#FROM continuumio/anaconda3:latest as intermediate
#
#
#WORKDIR /home/
#RUN git clone https://github.com/evancresswell/DCA_ER.git
#
#FROM continuumio/anaconda3:latest 
#
#RUN conda install biopython joblib scipy scikit-learn 
#RUN apt-get update
#RUN apt-get install -y curl grep sed dpkg
#
#RUN apt-get install -y libzmq3-dev python3-pip
#RUN apt-get install -y build-essential
#RUN apt-get clean && rm -rf /var/lib/apt/lists/*
#
#WORKDIR /home/DCA
#COPY --from=intermediate /home/DCA_ER/pydca /home/DCA_ER/pydca
#
#WORKDIR /home/DCA/biowulf
#RUN curl https://www.drive5.com/muscle/downloads3.8.31/muscle3.8.31_i86linux64.tar.gz | tar -xz
#RUN mv muscle3.8.31_i86linux64 muscle
#RUN chmod +x muscle
#
#RUN pip install --no-dependencies /home/DCA_ER/pydca/
#

# Python 3.7 version
FROM debian:buster-slim as intermediate
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

ENV PYTHON_VERSION 3.7

RUN apt-get update --fix-missing && \
    apt-get install -y wget build-essential bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion sudo && \
    apt-get clean

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# conda and test dependencies
RUN /opt/conda/bin/conda install -y -c defaults \
    conda conda-package-handling \
    python=$PYTHON_VERSION pycosat requests ruamel_yaml cytoolz \
    anaconda-client nbformat make \
    pytest pytest-cov codecov radon pytest-timeout mock responses pexpect \
    flake8 && \
    /opt/conda/bin/conda clean --all --yes

WORKDIR /home/
RUN git clone https://github.com/KIT-MBS/pydca.git

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

ENV PYTHON_VERSION 3.7

RUN apt-get update --fix-missing && \
    apt-get install -y wget build-essential bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion sudo && \
    apt-get clean

#RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh -O ~/anaconda.sh && \
#    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
#    rm ~/anaconda.sh && \
#    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
#    echo "conda activate base" >> ~/.bashrc && \
#    find /opt/conda/ -follow -type f -name '*.a' -delete && \
#    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
#    /opt/conda/bin/conda clean -afy
#
## conda and test dependencies
#RUN /opt/conda/bin/conda install -y -c defaults \
#    conda conda-package-handling \
#    python=$PYTHON_VERSION pycosat requests ruamel_yaml cytoolz \
#    anaconda-client nbformat make \
#    pytest pytest-cov codecov radon pytest-timeout mock responses pexpect \
#    flake8 && \
#    /opt/conda/bin/conda clean --all --yes
#

#WORKDIR /home/DCA/biowulf
#RUN curl https://www.drive5.com/muscle/downloads3.8.31/muscle3.8.31_i86linux64.tar.gz | tar -xz
#RUN mv muscle3.8.31_i86linux64 muscle
#RUN chmod +x muscle

RUN sudo /opt/conda/bin/pip install llvmlite --ignore-installed
RUN /opt/conda/bin/pip install --no-dependencies /home/pydca/
#RUN /opt/conda/bin/pip install /home/pydca/

#USER test_user

#ENTRYPOINT [ "/bin/bash" ]
CMD [ "/bin/bash" ]












### Old Version

#FROM ubuntu:latest as intermediate
## old anaconda declaration.. 
##ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
##ENV PATH /opt/conda/bin:$PATH
##
#RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
#    libglib2.0-0 libxext6 libsm6 libxrender1 \
#    git mercurial subversion
#
##RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh && \
##    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
##    rm ~/anaconda.sh && \
##    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
##    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
##    echo "conda activate base" >> ~/.bashrc
#
#RUN apt-get update
#RUN apt-get install -y curl grep sed dpkg 
#RUN apt-get install -y libzmq3-dev python3-pip
#RUN apt-get install build-essential
#RUN apt-get clean && rm -rf /var/lib/apt/lists/*
## Clone our private GitHub Repository
#WORKDIR /home
##RUN git clone https://4de5804a0bb1014bfacfc2466187741403f41b6c:x-oauth-basic@github.com/evancresswell/DCA_ER.git 
#RUN git clone https://github.com/evancresswell/DCA_ER.git
#
#RUN TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
#    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
#    dpkg -i tini.deb && \
#    rm tini.deb && \
#    apt-get clean
#
#
#FROM ubuntu:latest
#
##  $ docker build . -t continuumio/anaconda3:latest -t continuumio/anaconda3:5.3.0
##  $ docker run --rm -it continuumio/anaconda3:latest /bin/bash
##  $ docker push continuumio/anaconda3:latest
##  $ docker push continuumio/anaconda3:5.3.0
#
#ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
#ENV PATH /opt/conda/bin:$PATH
#
#RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
#    libglib2.0-0 libxext6 libsm6 libxrender1 \
#    git mercurial subversion
#
#RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh && \
#    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
#    rm ~/anaconda.sh && \
#    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
#    echo "conda activate base" >> ~/.bashrc
#
## Create the environment:
##COPY DCA_ER/er_images/environment.yml .
##RUN conda env create -f environment.yml
## Make RUN commands use the new environment:
##SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
#
#RUN apt-get update
#RUN apt-get install -y curl grep sed dpkg 
#RUN apt-get install -y libzmq3-dev python3-pip
#RUN apt-get install build-essential
#RUN apt-get clean && rm -rf /var/lib/apt/lists/*
#RUN pip install --upgrade pip
#RUN pip install -U jupyter jupyterlab
#RUN jupyter serverextension enable --py jupyterlab --sys-prefix
#ENV PATH /opt/conda/bin:$PATH
#RUN conda update conda && \
#    conda install biopython joblib scipy scikit-learn &&\ 
#    conda install numba --channel numba \ 
#
#WORKDIR /home/DCA_ER/
#COPY --from=intermediate /home/DCA_ER /home/DCA_ER 
#RUN pip install --no-dependencies /home/DCA_ER/pydca/
#
#RUN TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
#    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
#    dpkg -i tini.deb && \
#    rm tini.deb && \
#    apt-get clean
#
#ENTRYPOINT [ "/usr/bin/tini", "--" ]
#CMD [ "/bin/bash" ]
#
