FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    sudo \
    zsh \
    swig \
    libosmesa6-dev \
    patchelf \
    htop \
    && rm -rf /var/lib/apt/lists/*

RUN git config --global user.email "ryantse100@gmail.com"
RUN git config --global user.name "Ryan Tse"

# Neovim setup
RUN curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux64.tar.gz && \
    sudo rm -rf /opt/nvim && \
    sudo tar -C /opt -xzf nvim-linux64.tar.gz && \
    rm nvim-linux64.tar.gz
RUN echo 'export PATH="$PATH:/opt/nvim-linux64/bin"' >> /root/.zshrc
RUN git clone --depth 1 https://github.com/rytse/nvim.git \
    ~/.config/nvim
RUN git clone --depth 1 https://github.com/wbthomason/packer.nvim \
    ~/.local/share/nvim/site/pack/packer/start/packer.nvim

# Mamba setup
RUN wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
RUN bash Miniforge3.sh -b -p "${HOME}/conda"
RUN . "${HOME}/conda/etc/profile.d/conda.sh"
RUN . "${HOME}/conda/etc/profile.d/mamba.sh"
RUN rm Miniforge3.sh

# Mujoco setup
RUN mkdir ~/.mujoco
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
RUN tar -xzvf mujoco210-linux-x86_64.tar.gz  -C ~/.mujoco/
RUN rm mujoco210-linux-x86_64.tar.gz
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin" >> ~/.zshrc

# Clone repo
RUN git clone https://github.com/rytse/self-predictive-rl.git /workspace

# Set up python env
RUN mamba env create -f /workspace/mujoco_code/environment.yml
RUN mamba init
