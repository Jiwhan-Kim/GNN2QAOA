python gradient.py --environment simulator \
                   --shots 4096 \
                   --num-qubits 10 \
                   --num-layers 1 \
                   --draw-name "251228_sim_layer1" \
                   --lr 0.01 \
                   --beta1 0.9 \
                   --beta2 0.999 \
                   --eps 1e-8 \
                   --num-iterations 30 \
                   --print \
                   --draw \
                   --scheduler-type "onecycle" \
                   --max-lr 0.5 \
                   --total-steps 30 \
                   --pct-start 0.3 \
                   --div-factor 25.0 \
                   --final-div-factor 1e4


