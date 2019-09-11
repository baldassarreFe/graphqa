FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Common shell utils + oh-my-zsh
RUN apt-get update && \
    apt-get install -y git curl wget tree htop iotop zsh python3-pip tmux fonts-powerline && \
    pip3 install --no-cache-dir powerline-status && \
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)" && \
    sed 's/# DISABLE_AUTO_UPDATE="true"/DISABLE_AUTO_UPDATE="true"/' -i /root/.zshrc

# Zsh autosuggestions
RUN git clone https://github.com/zsh-users/zsh-autosuggestions /root/.oh-my-zsh/custom/plugins/zsh-autosuggestions  && \
    sed '/^plugins=(git)/ a plugins+=(zsh-autosuggestions)' -i /root/.zshrc

# Zsh syntax highlighting
RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git /root/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting && \
    sed '/^plugins=(git)/ a plugins+=(zsh-syntax-highlighting)' -i /root/.zshrc

# Zsh fuzzy finder (ctrl+r)
RUN git clone --depth 1 https://github.com/junegunn/fzf.git /root/.fzf && \
    /root/.fzf/install && \
    sed '/^plugins=(git)/ a plugins+=(fzf)' -i /root/.zshrc

SHELL ["zsh", "-c"]

# Miniconda + zsh autocompletions for conda
COPY .zshrc.conda /root/.zshrc.conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    sh miniconda.sh -b -p /root/miniconda && \
    rm miniconda.sh && \
    /root/miniconda/bin/conda init zsh && \
    git clone https://github.com/esc/conda-zsh-completion /root/.oh-my-zsh/custom/completions/conda-zsh-completion && \
    echo 'source /root/.zshrc.conda' >> /root/.zshrc

# Expose port for Jupyter lab
# EXPOSE 8888

# Expose port for Tensorboard
# EXPOSE 6006

# Install project-specific dependencies and aliases
COPY conda.yaml /conda.yaml
RUN source /root/.zshrc && \
    conda env update -n base --file /conda.yaml && \
    conda clean --quiet --yes --all

COPY .zshrc.addons /root/.zshrc.addons
RUN echo 'source /root/.zshrc.addons' >> /root/.zshrc

CMD zsh
