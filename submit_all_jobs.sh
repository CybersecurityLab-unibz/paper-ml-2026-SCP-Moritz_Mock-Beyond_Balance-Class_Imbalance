#!/bin/bash

# Define possible values for each parameter
LANGUAGES=("java" "pharo" "python")             
LOSSES=("EW" "ICF" "RBF")          
EPOCHS=(1 3 5 10)                       
BATCHSIZES=(1 2 4 8 16)                 
LRS=("3e-5" "4e-5" "5e-5")                
WDS=(0 0.001 0.0001)           

count=0

# Loop over all possible permutations
for LANGUAGE in "${LANGUAGES[@]}"; do
  for LOSS in "${LOSSES[@]}"; do
    for EPOCH in "${EPOCHS[@]}"; do
      for BATCHSIZE in "${BATCHSIZES[@]}"; do
        for LR in "${LRS[@]}"; do
          for WD in "${WDS[@]}"; do
            ((count++))
            echo "Submitting job #$count: LANGUAGE=$LANGUAGE, LOSS=$LOSS, EPOCH=$EPOCH, BATCHSIZE=$BATCHSIZE, LR=$LR, WD=$WD"
            sbatch cineca.batch "$LANGUAGE" "$LOSS" "$EPOCH" "$BATCHSIZE" "$LR" "$WD"

          done
        done
      done
    done
  done
done

echo "All job permutations submitted."
