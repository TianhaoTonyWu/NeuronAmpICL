#!/bin/bash

tasks=(
task242
task274
task1447
)

# Process data


# # # Get Activations

# for task in "${tasks[@]}"; do
#   python src/GV_activate.py -in "/root/few_vs_zero/data/sni/k1token/$task/tp4/train_1.pkl" -out "/root/few_vs_zero/data/sni/k1/$task/matrix/GV_last.json"  
#   python src/GV_activate.py -in "/root/few_vs_zero/data/sni/k3token/$task/tp4/train_3.pkl" -out "/root/few_vs_zero/data/sni/k3/$task/matrix/GV_last.json"  
#   python src/GV_activate.py -in "/root/few_vs_zero/data/sni/k7token/$task/tp4/train_7.pkl" -out "/root/few_vs_zero/data/sni/k7/$task/matrix/GV_last.json"  
#   python src/GV_activate.py -in "/root/few_vs_zero/data/sni/k9token/$task/tp4/train_9.pkl" -out "/root/few_vs_zero/data/sni/k9/$task/matrix/GV_last.json"  
# done


# ## filter and validation


pt_values=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5) 

for task in "${tasks[@]}"; do

  for pt in "${pt_values[@]}"; do

    python src/identify.py -pt "$pt" -in "/root/few_vs_zero/data/sni/k1/$task/matrix/GV_last.json" -out "/root/few_vs_zero/data/sni/k1/$task/activation_mask/GV_last/"
    python src/identify.py -pt "$pt" -in "/root/few_vs_zero/data/sni/k3/$task/matrix/GV_last.json" -out "/root/few_vs_zero/data/sni/k3/$task/activation_mask/GV_last/"
    python src/identify.py -pt "$pt" -in "/root/few_vs_zero/data/sni/k7/$task/matrix/GV_last.json" -out "/root/few_vs_zero/data/sni/k7/$task/activation_mask/GV_last/"
    python src/identify.py -pt "$pt" -in "/root/few_vs_zero/data/sni/k9/$task/matrix/GV_last.json" -out "/root/few_vs_zero/data/sni/k9/$task/activation_mask/GV_last/"
  done

  python src/validate_mask.py -in "/root/few_vs_zero/data/sni/k1token/$task/tp4/val_0.pkl" -mp "/root/few_vs_zero/data/sni/k1/$task/activation_mask/GV_last" -d 0 &
  python src/validate_mask.py -in "/root/few_vs_zero/data/sni/k3token/$task/tp4/val_0.pkl" -mp "/root/few_vs_zero/data/sni/k3/$task/activation_mask/GV_last" -d 1 &
  wait
  python src/validate_mask.py -in "/root/few_vs_zero/data/sni/k7token/$task/tp4/val_0.pkl" -mp "/root/few_vs_zero/data/sni/k7/$task/activation_mask/GV_last" -d 0 &
  python src/validate_mask.py -in "/root/few_vs_zero/data/sni/k9token/$task/tp4/val_0.pkl" -mp "/root/few_vs_zero/data/sni/k9/$task/activation_mask/GV_last" -d 1 &
  wait
done

