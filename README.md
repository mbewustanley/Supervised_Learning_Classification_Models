# Alpha-Gamma Predictions
## using multiple models

## Conda
```
conda create -n SLCenv python=3.11 -y

conda activate SLCenv

pip install -r requirements
```

## DVC
```
git init

dvc init

dvc repro
```

## Flask
```
pip install flask

# start backend
python main.py

# change directory to frontend folder
cd flask_api/frontend

python -m http.server 8000

# visit
http://localhost:8000




```