#!/bin/bash

tasks=(
task242
task274
task1447
)

# Process data


python src/process_sni.py -tp 4 -k 1 
python src/process_sni.py -tp 4 -k 3 
python src/process_sni.py -tp 4 -k 7 
python src/process_sni.py -tp 4 -k 9 




# # Get Activations

# for task in "${tasks[@]}"; do
#   python src/GV_activate.py -in "/home/wth/few_vs_zero/data/sni/token/$task/tp0/train_5.pkl" -out "/home/wth/few_vs_zero/data/sni/$task/matrix/GV.json"
#   python src/GV_activate.py -in "/home/wth/few_vs_zero/data/sni/token/$task/tp4/train_5.pkl" -out "/home/wth/few_vs_zero/data/sni/$task/matrix/GV_last.json"
#   python src/LAPE_activate.py -in "/home/wth/few_vs_zero/data/sni/token/$task/tp0/train_5.pkl" -out "/home/wth/few_vs_zero/data/sni/$task/matrix/LAPE.pth"

# done


# ## filter and validation


# pt_values=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5) 

# for task in "${tasks[@]}"; do

#   for pt in "${pt_values[@]}"; do

#     python src/identify.py -pt "$pt" -in "/home/wth/few_vs_zero/data/sni/$task/matrix/GV.json" -out "/home/wth/few_vs_zero/data/sni/$task/activation_mask/GV/"
#     python src/identify.py -pt "$pt" -in "/home/wth/few_vs_zero/data/sni/$task/matrix/GV_last.json" -out "/home/wth/few_vs_zero/data/sni/$task/activation_mask/GV_last/"
#     python src/identify.py -pt "$pt" -in "/home/wth/few_vs_zero/data/sni/$task/matrix/LAPE.pth" -out "/home/wth/few_vs_zero/data/sni/$task/activation_mask/LAPE/"

#   done

#   python src/validate_mask.py -in "/home/wth/few_vs_zero/data/sni/token/$task/tp0/val_0.pkl" -mp "/home/wth/few_vs_zero/data/sni/$task/activation_mask/GV" -d 0 &
#   python src/validate_mask.py -in "/home/wth/few_vs_zero/data/sni/token/$task/tp0/val_0.pkl" -mp "/home/wth/few_vs_zero/data/sni/$task/activation_mask/GV_last" -d 1 &
#   python src/validate_mask.py -in "/home/wth/few_vs_zero/data/sni/token/$task/tp0/val_0.pkl" -mp "/home/wth/few_vs_zero/data/sni/$task/activation_mask/LAPE" -d 2 &
#   wait


# done

