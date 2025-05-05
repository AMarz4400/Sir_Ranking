#!/bin/bash -l
#SBATCH --job-name=SisInf_FPRS_rep
#SBATCH --time=4-00:00:00                                   ## format: HH:MM:SS
#SBATCH --nodes=1
#SBATCH --mincpus=30
#SBATCH --mem=200GB                                       ## memory per node out of 494000MB (481GB)
#SBATCH --output=/leonardo/home/userexternal/amarzano/Sir_Ranking/slogs/SisInf_Sir_ranking_output-%A_%a.out
#SBATCH --error=/leonardo/home/userexternal/amarzano/Sir_Ranking/slogs/SisInf_Sir_ranking_error-%A_%a.err
#SBATCH --account=IscrC_MMMR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.marzanot@studenti.poliba.it
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod                                    ##    1 out of 4 or 8

module load profile/deeplrn
module load cuda/12.1
#module load gcc/12.2.0-cuda-12.1
module load python/3.10.8--gcc--11.3.0

source $HOME/Sir_Ranking/venv310/bin/activate

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd $HOME/Sir_Ranking

echo "Run experiments"

python pro_data/data_pro.py Toys_and_Games_5.json > logs/Toys_and_Games/data_pro.log  && python main.py train --model=DeepCoNN --num_fea=1 --output=fm --dataset=Toys_and_Games_data > logs/Toys_and_Games/2025-04-29-15_02_37.log 2>&1