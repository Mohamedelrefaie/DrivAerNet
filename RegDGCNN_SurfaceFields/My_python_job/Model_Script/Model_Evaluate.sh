python ./run_pipeline.py \
    --stages evaluate \
    --exp_name "Train_Test" \
    --dataset_path "$HOME/ML_Turbulent/Data_Pressure_Field/Data_Pressure/PressureVTK/N_S_WWS_WM" \
    --subset_dir "$HOME/ML_Turbulent/DrivAerNet/train_val_test_splits" \
    --cache_dir "$HOME/ML_Turbulent/Data_Pressure_Field/Cache_data" \
    --num_points 10000 \
    --num_eval_samples 5 \
    --gpus "0"
