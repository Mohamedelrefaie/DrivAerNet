python run_pipeline.py \
    --stages train \
    --exp_name "Train_Test" \
    --dataset_path "$HOME/Data_Pressure/Pressure_VTK" \
    --subset_dir "$HOME/ML_Turbulent/DrivAerNet/train_val_test_splits" \
    --cache_dir "$HOME/Data_Pressure/Cache_data" \
    --num_points 10000 \
    --num_workers 1 \
    --batch_size 8 \
    --epochs 12 \
    --gpus "0"


