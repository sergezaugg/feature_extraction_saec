python -m venv ./

pip uninstall fe_saec 

pip install dist/fe_saec-0.9.6-py3-none-any.whl
pip install https://github.com/sergezaugg/feature_extraction_saec/releases/download/v0.9.3/fe_saec-0.9.3-py3-none-any.whl

pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126


# need to install this order 
pip install torch torchvision
pip install -r requirements.txt

