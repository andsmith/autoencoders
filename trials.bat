# 2-layer

python .\sparse.py --layers  64 1024 --epochs 100 --stages 40  --no_plot --reg_lambda .01
python .\sparse.py --layers 128 1024 --epochs 100 --stages 40  --no_plot --reg_lambda .01
python .\sparse.py --layers 256 1024 --epochs 100 --stages 40  --no_plot --reg_lambda .01
python .\sparse.py --layers 512 1024 --epochs 100 --stages 40  --no_plot --reg_lambda .01

python .\sparse.py --layers  64 1024 --epochs 100 --stages 40  --no_plot --reg_lambda .01 --reg_method entropy --real_code_activations
python .\sparse.py --layers 128 1024 --epochs 100 --stages 40  --no_plot --reg_lambda .01 --reg_method entropy --real_code_activations
python .\sparse.py --layers 256 1024 --epochs 100 --stages 40  --no_plot --reg_lambda .01 --reg_method entropy --real_code_activations
python .\sparse.py --layers 512 1024 --epochs 100 --stages 40  --no_plot --reg_lambda .01 --reg_method entropy --real_code_activations


python .\sparse.py --layers 256 64 1024 --epochs 100 --stages 40  --no_plot --reg_lambda .01 --reg_method entropy --real_code_activations
python .\sparse.py --layers 256 128 1024 --epochs 100 --stages 40  --no_plot --reg_lambda .01 --reg_method entropy --real_code_activations
python .\sparse.py --layers 256 256 1024 --epochs 100 --stages 40  --no_plot --reg_lambda .01 --reg_method entropy --real_code_activations
python .\sparse.py --layers 256 512 1024 --epochs 100 --stages 40  --no_plot --reg_lambda .01 --reg_method entropy --real_code_activations

