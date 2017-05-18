#! /bin/bash

## Settings
inputfile='/home/axel/data/small/obs_2016-12_USDT_BTC_maxVol100.dict'

experiment_name='USDTBTC_Qtable_100vol10_60T4_Dez'
outputfolder='experiments_baselineMarket_costsFixed'
outfile_agent=$outputfolder"/"$experiment_name"/"$experiment_name".json"
outfile_samples=$outputfolder"/"$experiment_name"/"$experiment_name".csv"

volume=100
volume_intervals=10
decision_points=4
period_length=15
actions=[-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]
lim_stepsize=0.1

state_variables=['volume','time']
limit_base='incStepUnits'

echo "Starting Experiment '"$experiment_name"'"
echo "Inputfile for training: "$inputfile
echo "Outputfile for agent: "$outfile_agent
echo "Outputfile for samples: "$outfile_samples

if [ -d $outputfolder/$experiment_name ]; then
  echo "Found an existing folder '"$outputfolder/$experiment_name"'"

  read -p "Overwrite (y/n)? " answer
  case ${answer:0:1} in
      y|Y ) echo "Overwriting any existing files";;
      * )   echo "Aborted!"; exit;;
  esac
fi

mkdir $outputfolder/$experiment_name
cp run_training.sh $outputfolder/$experiment_name/
cp train_QTable.py $outputfolder/$experiment_name/

echo "Inputfile: "$inputfile
/home/axel/anaconda3/bin/python train_QTable.py \
					   --inputfile=$inputfile \
					   --volume=$volume \
					   --volume_intervals=$volume_intervals \
					   --decision_points=$decision_points \
					   --period_length=$period_length \
					   --actions=$actions \
					   --lim_stepsize=$lim_stepsize \
					   --state_variables=$state_variables \
					   --limit_base=$limit_base \
					   --outfile_agent=$outfile_agent \
					   --outfile_samples=$outfile_samples #>& $outputfolder/$experiment_name/'log.txt
