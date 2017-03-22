#! /bin/bash

## Settings
experiment_name='1701_USDTBTC_Qtable_100vol10_60T4'
inputfile='/home/axel/data/obs_2017-01_USDT_BTC_range1.2.dict'
volume=100
volume_intervals=10
decision_points=4
periodlength=15
action_min=-0.4
action_max=1.0
action_count=15

folder='/home/axel/notebooks/orderbook_agent/1_version2/Runs/experiments'
plotQ=True
state_variables=['volume','time']

echo "Starting Experiment '"$experiment_name"'"
echo "Inputfile for training: "$inputfile

if [ -d $folder/$experiment_name ]; then
  echo "Found an existing folder '"$folder/$experiment_name"'"

  read -p "Overwrite (y/n)? " answer
  case ${answer:0:1} in
      y|Y ) echo "Overwriting any existing files";;
      * )   echo "Aborted!"; exit;;
  esac
fi

mkdir -p $folder/$experiment_name/'model'
mkdir -p $folder/$experiment_name/'graphs'

cp run_training.sh $folder/$experiment_name/
cp train_Qtable.py $folder/$experiment_name/

python train_Qtable.py --experiment_name=$experiment_name --inputfile=$inputfile --volume=$volume --volume_intervals=$volume_intervals --decision_points=$decision_points --periodlength=$periodlength --action_min=$action_min --action_max=$action_max --action_count=$action_count --folder=$folder --plotQ=$plotQ --state_variables=$state_variables >& $folder/$experiment_name/'log.txt'
