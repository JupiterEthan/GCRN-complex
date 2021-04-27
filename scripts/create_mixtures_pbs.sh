#PBS -A PAA0005
#PBS -N gen_mix
#PBS -l walltime=5:00:00
#PBS -l nodes=1:ppn=20
#PBS -m abe
#PBS -o log_gen_mix.out
#PBS -e log_gen_mix.err

cd $PBS_O_WORKDIR/

python -B ./utils/gen_mix.py --mode=tt

python -B ./utils/gen_mix.py --mode=cv

python -B ./utils/gen_mix.py --mode=tr
