# Install pytorch

-- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

python -m venv .venv
source .venv/bin/activate
deactivate
pip install torch torchvision torchaudio
pip freeze > requirements.txt
pip install -r requirements.txt