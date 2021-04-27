#PBS -A PAA0005
#PBS -N GCRN
#PBS -l walltime=96:00:00
#PBS -l nodes=1:ppn=2:gpus=1
#PBS -m abe
#PBS -o log_train.out
#PBS -e log_train.err

module load cuda/10.2.89

cd $PBS_O_WORKDIR/

ckpt_dir=exp
gpus=0

CUDA_LAUNCH_BLOCKING=1

python -B ./train.py \
    --gpu_ids=$gpus \
    --tr_list=../filelists/tr_list.txt \
    --cv_file=../data/datasets/cv/cv.ex \
    --ckpt_dir=$ckpt_dir \
    --logging_period=1000 \
    --clip_norm=5.0 \
    --lr=0.001 \
    --time_log=./time.log \
    --unit=utt \
    --batch_size=8 \
    --buffer_size=16 \
    --max_n_epochs=30 \
    --resume_model=./exp/models/latest.pt \
