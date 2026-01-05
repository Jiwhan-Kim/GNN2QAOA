python test_get_thetas.py --environment qpu \
                   --shots 2048 \
                   --num-qubits 6 \
                   --num-layers 2 \
                   --draw-name "260103_qpu_increasing_shot" \
                   --lr 0.1 \
                   --beta1 0.9 \
                   --beta2 0.999 \
                   --eps 1e-8 \
                   --num-iterations 30 \
                   --print \
                   --draw \
                   --scheduler-type "onecycle" \
                   --max-lr 0.25 \
                   --total-steps 30 \
                   --pct-start 0.3 \
                   --div-factor 25.0 \
                   --final-div-factor 1e4


