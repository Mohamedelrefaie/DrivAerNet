python run_pipeline.py \
    --stages train \
    --exp_name "DrivAerNet_Pressure" \
    --dataset_path "$HOME/ML_Turbulent/Data_Pressure_Field/Data_Pressure/PressureVTK/N_S_WWS_WM" \
    --subset_dir "$HOME/ML_Turbulent/DrivAerNet/train_val_test_splits" \
    --cache_dir "$HOME/ML_Turbulent/Data_Pressure_Field/Cache_data" \
    --num_points 10000 \
    --batch_size 12 \
    --epochs 150 \
    --gpus "0"


# === Variable  ===
# --batch_size 
# It defines how many samples are processed at once per training step
#
# --epochs
# 
# Total number of training rounds over the whole dataset
# e.g. 10000 samples, batch_size = 100
#      10 000 / 100 = 100 steps for one epoch
#      --epochs 150 have 150 times loop
