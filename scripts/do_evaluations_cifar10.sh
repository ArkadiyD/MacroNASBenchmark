CUDA_VISIBLE_DEVICES=0 python3 src/main.py --dir=benchmark_cifar10 --data_path=datasets --dataset=CIFAR10 --training=0 --batch_size=128 --num_workers=4 --first_net_id=0 --last_net_id=1000
