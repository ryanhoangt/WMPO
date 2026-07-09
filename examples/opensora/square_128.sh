GPUS_PER_NODE=1
MASTER_ADDR=localhost:29522
NNODES=1
RANK=0



CONFIG=${1:-"checkpoint_files/world_models/square/P_128/train_mimicgen_cfg.py"}

torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --rdzv_endpoint $MASTER_ADDR \
    --rdzv_backend c10d \
    dependencies/opensora/scripts/train_web.py \
    $CONFIG \
