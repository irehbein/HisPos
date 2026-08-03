# paper-stance-detection-german-parliament

conda create -n poscor python=3.12

conda activate poscor

pip install -U transformers torch kernels 

pip install pytest

pip install gpt-oss

NEXT: python -m gpt_oss.chat model/



#####

python -m gpt_oss.chat model/


pip uninstall triton
pip install triton triton_kernels

Successfully installed triton-3.6.0 triton_kernels-0.1.0

# doesn't work, fix with:
pip uninstall triton_kernels -y

pip install git+https://github.com/triton-lang/triton.git


