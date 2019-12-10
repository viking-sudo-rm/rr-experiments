#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%j.out
#SBATCH --gpus=1

source /home/willm/anaconda3/bin/activate allennlp
conf=/home/willm/code/rr-experiments/configs/prefix.jsonnet

rnn=qrnn+
layers=1

dir=/tmp/willm/pre-$rnn
RNN=$rnn LAYERS=$layers allennlp train $conf -s $dir --include-package=rr_experiments
