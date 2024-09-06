# Install pytorch

-- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

python -m venv .venv
source .venv/bin/activate
deactivate
pip install torch torchvision torchaudio pyinstaller
pip freeze > requirements.txt
pip install -r requirements.txt

# Series of commands

On Colab:
train.py
python evaluate.py

# Script the model (converts it into a format suitable for deployment)
python torch_script.py
python torch_script_quantized.py

## On Local:
python inference.py
python run_inference.py
python run_inference_pt.py
python run_inference_pt_percentages.py ./data/20240831190201.jpg
python run_inference_pt_percentages.py ./data/20240902082401.jpg
python run_inference_pt_percentages.py /home/pi/Projects/parking_lot_classifier/scripts/ftp/last.jpg
rm -rf ./build && cp ./models/v2/car_detection_cnn_scripted_quantized.pt ./models/prod/model.pt && pyinstaller car_detection_cnn.spec && cp dist/car_detection ../astro/db

