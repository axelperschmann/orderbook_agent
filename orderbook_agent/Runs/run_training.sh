#! /bin/bash

## Settings
inputfile='/home/axel/data/small/obs_2016-11_USDT_BTC_maxVol100.dict'

experiment_name='3Month_USDTBTC_Qtable_100vol10_60T4'
outputfolder='experiments'
outputfile_model=$outputfolder"/"$experiment_name"/"$experiment_name".json"

volume=100
volume_intervals=10
decision_points=4
period_length=15
action_min=-0.4
action_max=1.0
action_count=15

state_variables=['volume','time']

echo "Starting Experiment '"$experiment_name"'"
echo "Inputfile for training: "$inputfile
echo "Outputfile for model: "$outputfile_model

if [ -d $folder/$experiment_name ]; then
  echo "Found an existing folder '"$folder/$experiment_name"'"

  read -p "Overwrite (y/n)? " answer
  case ${answer:0:1} in
      y|Y ) echo "Overwriting any existing files";;
      * )   echo "Aborted!"; exit;;
  esac
fi
mkdir $folder/$experiment_name
cp run_training.sh $folder/$experiment_name/
cp train_QTable.py $folder/$experiment_name/

echo $inputfile
python train_QTable.py --inputfile=$inputfile \
					   --volume=$volume \
					   --volume_intervals=$volume_intervals \
					   --decision_points=$decision_points \
					   --period_length=$period_length \
					   --action_min=$action_min \
					   --action_max=$action_max \
					   --action_count=$action_count \
					   --state_variables=$state_variables \
					   --outputfile_model=$outputfile_model #>& $folder/$experiment_name/'log.txt
