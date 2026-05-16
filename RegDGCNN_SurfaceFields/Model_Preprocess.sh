python run_pipeline.py \
    --stages preprocess \
    --exp_name "DrivAerNet_Pressure" \
    --dataset_path "$HOME/ML_Turbulent/Pressure_Field/Data_Pressure/PressureVTK/N_S_WWS_WM" \
    --cache_dir "$HOME/ML_Turbulent/Pressure_Field/Cache_data" \
    --subset_dir "$HOME/ML_Turbulent/Pressure_Field/train_val_test_splits" \
    --num_points 10000
