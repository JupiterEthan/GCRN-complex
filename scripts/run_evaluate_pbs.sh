#PBS -A PAA0005
#PBS -N GCRN_g2
#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=2
#PBS -m abe
#PBS -o log_eval.out
#PBS -e log_eval.err

module load cuda/10.2.89

cd $PBS_O_WORKDIR/

test_step=false
eval_step=true

ckpt_dir=exp
gpus=0

if $test_step; then
    python -B ./test.py \
        --gpu_ids=$gpus \
        --tt_list=../filelists/tt_list.txt \
        --ckpt_dir=$ckpt_dir \
        --model_file=./${ckpt_dir}/models/best.pt
fi

if $eval_step; then
    python -B ./measure.py --metric=stoi --tt_list=../filelists/tt_list.txt --ckpt_dir=$ckpt_dir
    python -B ./measure.py --metric=pesq --tt_list=../filelists/tt_list.txt --ckpt_dir=$ckpt_dir
    python -B ./measure.py --metric=snr --tt_list=../filelists/tt_list.txt --ckpt_dir=$ckpt_dir
fi
