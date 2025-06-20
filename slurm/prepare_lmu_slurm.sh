wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/bin/activate
conda create -n py310activerlhf python=3.10
conda activate py310activerlhf
conda install python=3.10
conda install conda-forge::xvfbwrapper
conda install conda-forge::ffmpeg
python3.10 -m pip install --user pipx
python3.10 -m pipx ensurepath
python3.10 -m pipx install poetry

mkdir -p ~/work/aiml
cd ~/work/aiml
git clone git@github.com:bkuen/active-rlhf.git

cd active-rlhf
poetry install