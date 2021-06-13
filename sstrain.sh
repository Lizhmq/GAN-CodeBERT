export CUDA_VISIBLE_DEVICES=0
DATADIR=./data/
OUTPUTDIR=./save/ss3-1
PRETRAINDIR=microsoft/codebert-base
LOGFILE=ss3-1.log
PER_NODE_GPU=1
PER_GPU_BATCH_TRAIN=14
PER_GPU_BATCH_EVAL=28
GRAD_ACC=2
EPOCH=20
BLOCKSIZE=512

# -m torch.distributed.launch --nproc_per_node=$PER_NODE_GPU
python  sstrain.py \
        --data_dir=$DATADIR \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --block_size=$BLOCKSIZE \
        --do_train \
        --node_index 0 \
        --gpu_per_node $PER_NODE_GPU \
        --learning_rate=4e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=$PER_GPU_BATCH_TRAIN \
        --per_gpu_eval_batch_size=$PER_GPU_BATCH_EVAL \
        --gradient_accumulation_steps=$GRAD_ACC \
        --num_train_epochs=$EPOCH \
        --logging_steps=50 \
        --save_steps=600 \
        --overwrite_output_dir \
        --seed=2233 \
        --train_name trainss3.pkl