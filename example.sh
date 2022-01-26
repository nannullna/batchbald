# python run.py --run_name exp1_resnet18 --dataset cifar10 --device cuda:0 --max_epochs 100 --batch_size 512 \
#  --early_stopping_threshold 0.95 --query_type random

python run.py --seed 42 --run_name exp3_resnet18 --dataset cifar10 --device cuda:1 \
    --max_epochs 250 --batch_size 512 --early_stopping_threshold 0.90 \
    --query_size 500 --query_type bald --initial_label_rate 0.1 \
    --wandb_project al_size_500_3
