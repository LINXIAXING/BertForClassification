export OMP_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1  #指定显卡
python launch.py "init_folder"
torchrun \
  --nproc_per_node 2 \
  --master_port 29501 \
  launch.py "train_multi_gpu"
#  "$@"