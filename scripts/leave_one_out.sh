#!/bin/bash

# Define the list of variables
list=("temperature_2m" "dewpoint_temperature_2m" "surface_pressure")

# Function to generate combinations
generate_combinations() {
  local arr=("$@")
  local len=${#arr[@]}
  local combinations=""
  for ((i=0; i<$len; i++)); do
    for ((j=i+1; j<$len; j++)); do
      combinations+="[\"${arr[i]}\",\"${arr[j]}\"]"
      if [[ $i -lt $((len-2)) || $j -lt $((len-1)) ]]; then
        combinations+=","
      fi
    done
  done
  echo "$combinations"
}

# Generate combinations
combinations=$(generate_combinations "${list[@]}")

python src/train.py --multirun data.species="Common Buzzard" data.era5_main_variables="$combinations"