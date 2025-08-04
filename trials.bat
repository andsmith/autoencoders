# 2-layer

python .\sparse.py --layers 64 4096 --epochs 25 --stages 40  --no_plot  --reg_lambda .02
python .\sparse.py --layers 64 4096 --epochs 25 --stages 40  --no_plot  --reg_lambda .002
python .\sparse.py --layers 128 4096 --epochs 25 --stages 40  --no_plot  --reg_lambda .02
python .\sparse.py --layers 128 4096 --epochs 25 --stages 40  --no_plot  --reg_lambda .002
python .\sparse.py --layers 256 4096 --epochs 25 --stages 40  --no_plot --reg_lambda .02
python .\sparse.py --layers 256 4096 --epochs 25 --stages 40  --no_plot --reg_lambda .002
python .\sparse.py --layers 512 4096 --epochs 25 --stages 40  --no_plot --reg_lambda .02
python .\sparse.py --layers 512 4096 --epochs 25 --stages 40  --no_plot --reg_lambda .002

# 3-layer

python .\sparse.py --layers 256 64 1024 --epochs 25 --stages 50  --no_plot  --reg_lambda .01
python .\sparse.py --layers 256 64 1024 --epochs 25 --stages 50  --no_plot  --reg_lambda .001
python .\sparse.py --layers 256 64 4096 --epochs 25 --stages 40  --no_plot  --reg_lambda .01
python .\sparse.py --layers 256 64 4096 --epochs 25 --stages 50  --no_plot  --reg_lambda .001

python .\sparse.py --layers 256 128 1024 --epochs 25 --stages 50  --no_plot  --reg_lambda .01
python .\sparse.py --layers 256 128 1024 --epochs 25 --stages 50  --no_plot  --reg_lambda .001
python .\sparse.py --layers 256 128 4096 --epochs 25 --stages 40  --no_plot  --reg_lambda .01
python .\sparse.py --layers 256 128 4096 --epochs 25 --stages 50  --no_plot  --reg_lambda .001

python .\sparse.py --layers 256 256 1024 --epochs 25 --stages 50  --no_plot --reg_lambda .01
python .\sparse.py --layers 256 256 1024 --epochs 25 --stages 50  --no_plot --reg_lambda .001
python .\sparse.py --layers 256 256 4096 --epochs 25 --stages 40  --no_plot --reg_lambda .01
python .\sparse.py --layers 256 256 4096 --epochs 25 --stages 50  --no_plot --reg_lambda .001

python .\sparse.py --layers 256 512 1024 --epochs 25 --stages 50  --no_plot --reg_lambda .01
python .\sparse.py --layers 256 512 1024 --epochs 25 --stages 50  --no_plot --reg_lambda .001
python .\sparse.py --layers 256 512 4096 --epochs 25 --stages 40  --no_plot --reg_lambda .01
python .\sparse.py --layers 256 512 4096 --epochs 25 --stages 50  --no_plot --reg_lambda .001



# To what extent do middle layer size, code size, and regularization constant affect MSE?

