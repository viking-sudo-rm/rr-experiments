#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/slurm/%j.out
#SBATCH --gpus=1

source /home/willm/anaconda3/bin/activate
conf=/home/willm/code/rr-experiments/configs/max_difference.jsonnet

rnn=lstm
dir=/tmp/willm/$rnn

RNN=$rnn allennlp train $conf -s $dir --include-package=rr_experiments
