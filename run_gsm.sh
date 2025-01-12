#!/bin/bash


# ACTIVATE

# python src/GV_activate.py -in "/home/wth/few_vs_zero/data/gsm/token0/train_5.pkl" -out "/home/wth/few_vs_zero/data/gsm/matrix/GV.json"
# python src/GV_activate.py -in "/home/wth/few_vs_zero/data/gsm/token4/train_5.pkl" -out "/home/wth/few_vs_zero/data/gsm/matrix/GV_last.json"
# python src/LAPE_activate.py -in "/home/wth/few_vs_zero/data/gsm/token0/train_5.pkl" -out "/home/wth/few_vs_zero/data/gsm/matrix/LAPE.pth"

# IDENTIFY

# pt_values=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5) 
# pt_values=(0.01 0.02 0.03 0.04) 
# for pt in "${pt_values[@]}"; do
#     python src/identify.py -in "/home/wth/few_vs_zero/data/gsm/matrix/GV.json" -out "/home/wth/few_vs_zero/data/gsm/activation_mask/GV/" -pt $pt
#     python src/identify.py -in "/home/wth/few_vs_zero/data/gsm/matrix/GV_last.json" -out "/home/wth/few_vs_zero/data/gsm/activation_mask/GV_last/" -pt $pt
#     python src/identify.py -in "/home/wth/few_vs_zero/data/gsm/matrix/LAPE.pth" -out "/home/wth/few_vs_zero/data/gsm/activation_mask/LAPE/" -pt $pt
# done

# # VALIDATE

# python src/validate_mask.py -in "/home/wth/few_vs_zero/data/gsm/token0/val_0.pkl" -mp "/home/wth/few_vs_zero/data/gsm/activation_mask/GV" -d 0 &
# python src/validate_mask.py -in "/home/wth/few_vs_zero/data/gsm/token0/val_0.pkl" -mp "/home/wth/few_vs_zero/data/gsm/activation_mask/GV_last" -d 1 &
# python src/validate_mask.py -in "/home/wth/few_vs_zero/data/gsm/token0/val_0.pkl" -mp "/home/wth/few_vs_zero/data/gsm/activation_mask/LAPE" -d 2 &
# wait

# EVALUATE

# python src/evaluate.py -mask /home/wth/few_vs_zero/data/gsm/activation_mask/GV/0.05p.pth -in "/home/wth/few_vs_zero/data/gsm/token0/test_0.pkl" -x 1.3 -d "0" &
# python src/evaluate.py -mask /home/wth/few_vs_zero/data/gsm/activation_mask/LAPE/0.05p.pth -in "/home/wth/few_vs_zero/data/gsm/token0/test_0.pkl" -x 1.5 -d "1" &
# python src/evaluate.py -mask /home/wth/few_vs_zero/data/gsm/activation_mask/GV_last/0.25p.pth -in "/home/wth/few_vs_zero/data/gsm/token0/test_0.pkl" -x 1.2 -d "2" &
# python src/evaluate.py -in "/home/wth/few_vs_zero/data/gsm/token0/test_0.pkl" -x 1.0 -d "3" &
# python src/evaluate.py -in "/home/wth/few_vs_zero/data/gsm/token0/test_5.pkl" -x 1.0 -d "4" &
# wait

python src/identify.py --random --out "/home/wth/few_vs_zero/data/gsm/activation_mask/Random/" --seed 42
python src/identify.py --random --out "/home/wth/few_vs_zero/data/gsm/activation_mask/Random/" --seed 1
python src/identify.py --random --out "/home/wth/few_vs_zero/data/gsm/activation_mask/Random/" --seed 13
python src/evaluate.py -in "/home/wth/few_vs_zero/data/gsm/token0/test_0.pkl" -x 1.7 -mask  "/home/wth/few_vs_zero/data/gsm/activation_mask/Random/random1.pth" -d 0 &
python src/evaluate.py -in "/home/wth/few_vs_zero/data/gsm/token0/test_0.pkl" -x 1.7 -mask  "/home/wth/few_vs_zero/data/gsm/activation_mask/Random/random42.pth" -d 1 &
python src/evaluate.py -in "/home/wth/few_vs_zero/data/gsm/token0/test_0.pkl" -x 1.7 -mask  "/home/wth/few_vs_zero/data/gsm/activation_mask/Random/random13.pth" -d 2 &
wait




