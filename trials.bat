# 2-layer
python .\sparse.py --layers  64 1024 --epochs 100 --stages 10  --no_plot --reg_lambda .01 --reg_method entropy
python .\sparse.py --layers  64 2048 --epochs 100 --stages 10  --no_plot --reg_lambda .01 --reg_method entropy
python .\sparse.py --layers  64 4096 --epochs 100 --stages 10  --no_plot --reg_lambda .01 --reg_method entropy

python .\sparse.py --layers  64 1024 --epochs 100 --stages 10  --no_plot --reg_lambda .01
python .\sparse.py --layers 128 1024 --epochs 100 --stages 10  --no_plot --reg_lambda .01
python .\sparse.py --layers 256 1024 --epochs 100 --stages 10  --no_plot --reg_lambda .01
python .\sparse.py --layers 512 1024 --epochs 100 --stages 10  --no_plot --reg_lambda .01

python .\sparse.py --layers  64 1024 --epochs 100 --stages 10  --no_plot --reg_lambda .01 --reg_method entropy
python .\sparse.py --layers 128 1024 --epochs 100 --stages 10  --no_plot --reg_lambda .01 --reg_method entropy
python .\sparse.py --layers 256 1024 --epochs 100 --stages 10  --no_plot --reg_lambda .01 --reg_method entropy
python .\sparse.py --layers 512 1024 --epochs 100 --stages 10  --no_plot --reg_lambda .01 --reg_method entropy

