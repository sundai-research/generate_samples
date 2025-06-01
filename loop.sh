#! /bin/bash

# Create output directory
mkdir -p output

# Loop through each JSONL file in data and generate samples
for file in data/*.jsonl; do
  base=$(basename "$file" .jsonl)
  cmd="python generate_samples.py --input-path $file --output-path output/${base}_output.jsonl --temperature 1.0 --num-responses-per-question 256"
  echo $cmd
  $cmd
done

