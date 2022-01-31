python run.py --seed 3027 --run_name resnet18_es0.95 --dataset cifar10 --device cuda:0 \
  --max_epochs 500 --batch_size 256 --early_stopping_threshold 0.95 \
  --query_size 500 --query_type kmeans --initial_label_ratio 0.1 --eval_ratio 0.1 \
  --log_every 5 --wandb_project al_baselines --tolerance 10 --eval_every 5 \
  --optimizer_type adam --learning_rate 0.001
