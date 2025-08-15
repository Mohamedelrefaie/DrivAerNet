python run_pipeline.py \
    --stages preprocess \
    --exp_name "Train_Test" \
    --dataset_path "$HOME/ML_Turbulent/Data_Pressure_Field/Data_Pressure/PressureVTK/N_S_WWS_WM" \
    --subset_dir "$HOME/ML_Turbulent/DrivAerNet/train_val_test_splits" \
    --cache_dir "$HOME/ML_Turbulent/Data_Pressure_Field/Cache_data" \
    --num_points 50000

#    --dataset_path "$HOME/Data_Pressure/Pressure_VTK" \
#    --subset_dir "$HOME/ML_Turbulent/DrivAerNet/train_val_test_splits" \
#    --cache_dir "$HOME/Data_Pressure/Cache_data" \

#    --dataset_path "$HOME/ML_Turbulent/Data_Pressure_Field/Data_Pressure/PressureVTK/N_S_WWS_WM" \
#    --subset_dir "$HOME/ML_Turbulent/DrivAerNet/train_val_test_splits" \
#    --cache_dir "$HOME/ML_Turbulent/Data_Pressure_Field/Cache_data" \
