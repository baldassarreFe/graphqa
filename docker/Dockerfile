#FROM nvcr.io/nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
FROM nvcr.io/nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update && \
    apt-get install -y git curl wget tree htop zsh python3-pip screen && \
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"

RUN git clone https://github.com/zsh-users/zsh-autosuggestions /root/.oh-my-zsh/custom/plugins/zsh-autosuggestions  && \
    sed '/^plugins=(git)/ a plugins+=(zsh-autosuggestions)' -i /root/.zshrc

RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git /root/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting && \
    sed '/^plugins=(git)/ a plugins+=(zsh-syntax-highlighting)' -i /root/.zshrc

RUN FZF_BASE="${HOME}/.fzf" && \
    git clone --depth 1 https://github.com/junegunn/fzf.git "${FZF_BASE}" && "${FZF_BASE}"/install && \
    sed '/^plugins=(git)/ a plugins+=(fzf)' -i /root/.zshrc

RUN pip3 install powerline-status && \
    apt-get -y install fonts-powerline

COPY .zshrc.addons /root/.zshrc.addons
RUN echo 'source /root/.zshrc.addons' >> /root/.zshrc

SHELL ["zsh", "-c"]

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    sh miniconda.sh -b -p /root/miniconda && \
    rm miniconda.sh && \
    /root/miniconda/bin/conda init zsh

RUN source /root/.zshrc && \
    conda activate base && \
    conda install python=3.7 numpy matplotlib pandas jupyterlab jupyter_console tqdm scipy scikit-learn && \
    conda install pytorch cudatoolkit=10.0 -c pytorch && \
    conda clean --all

# Jupyter lab
EXPOSE 8888

COPY conda.yaml /conda.yaml
RUN source /root/.zshrc && \
    conda env update -n base --file /conda.yaml && \
    conda clean --all

# Tensorboard
EXPOSE 6006

COPY torchgraphs /torchgraphs
RUN source /root/.zshrc && \
    conda activate base && \
    pip install /torchgraphs

CMD zsh
