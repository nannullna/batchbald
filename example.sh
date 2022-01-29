# python run.py --run_name exp1_resnet18 --dataset cifar10 --device cuda:0 --max_epochs 100 --batch_size 512 \
#  --early_stopping_threshold 0.95 --query_type random

python run.py --seed 1528 --run_name resnet18_es0.95 --dataset cifar10 --device cuda:0 \
  --max_epochs 1 --batch_size 256 --early_stopping_threshold 0.9 \
  --query_size 500 --query_type kmeans --initial_label_rate 0.1 \
  --log_every 50 --wandb_project al_baselines