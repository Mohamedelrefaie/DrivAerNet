python ../run_pipeline.py \
    --stages train \
    --exp_name "Train_Test" \
    --test_only  \
    --num_points 10000 \
    --num_workers 1 \
    --dataset_path "$HOME/ML_Turbulent/Data_Pressure_Field/Data_Pressure/PressureVTK/N_S_WWS_WM" \
    --subset_dir "$HOME/ML_Turbulent/DrivAerNet/train_val_test_splits" \
    --cache_dir "$HOME/ML_Turbulent/Data_Pressure_Field/Cache_data" \
    --batch_size 6 \
    --epochs 150 \
    --gpus "0"



#    --dataset_path "$HOME/Data_Pressure/Pressure_VTK" \
#    --subset_dir "$HOME/ML_Turbulent/DrivAerNet/train_val_test_splits" \
#    --cache_dir "$HOME/Data_Pressure/Cache_data" \

#    --dataset_path "$HOME/ML_Turbulent/Data_Pressure_Field/Data_Pressure/PressureVTK/N_S_WWS_WM" \
#    --subset_dir "$HOME/ML_Turbulent/DrivAerNet/train_val_test_splits" \
#    --cache_dir "$HOME/ML_Turbulent/Data_Pressure_Field/Cache_data" \
