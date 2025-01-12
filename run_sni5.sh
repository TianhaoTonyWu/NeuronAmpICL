
#!/bin/bash

tasks=(
task242
task274
task1447
task403
task645
task475
)
for task in "${tasks[@]}"; do
    # python src/identify.py --random --out "/home/wth/few_vs_zero/data/sni/$task/activation_mask/Random/" --seed 42
    # python src/identify.py --random --out "/home/wth/few_vs_zero/data/sni/$task/activation_mask/Random/" --seed 1
    # python src/identify.py --random --out "/home/wth/few_vs_zero/data/sni/$task/activation_mask/Random/" --seed 13

    python src/evaluate.py -in "/home/wth/few_vs_zero/data/sni/token/$task/tp0/test_0.pkl" -x 1.7 -mask  "/home/wth/few_vs_zero/data/sni/$task/activation_mask/Random/random1.pth" -d 0 &
    python src/evaluate.py -in "/home/wth/few_vs_zero/data/sni/token/$task/tp0/test_0.pkl" -x 1.7 -mask  "/home/wth/few_vs_zero/data/sni/$task/activation_mask/Random/random42.pth" -d 1 &
    python src/evaluate.py -in "/home/wth/few_vs_zero/data/sni/token/$task/tp0/test_0.pkl" -x 1.7 -mask  "/home/wth/few_vs_zero/data/sni/$task/activation_mask/Random/random13.pth" -d 2 &
    wait
done