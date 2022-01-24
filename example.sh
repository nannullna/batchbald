# python run.py --run_name exp1_resnet18 --dataset cifar10 --device cuda:0 --max_epochs 100 --batch_size 512 \
#  --early_stopping_threshold 0.95 --query_type random

python run.py --seed 42 --run_name exp2_resnet18 --dataset cifar10 --device cuda:0 \
    --max_epochs 50 --batch_size 128 --early_stopping_threshold 0.95 \
    --query_size 200 --query_type bald --initial_label_rate 0.1 \
    --wandb_project active_learning_size_100