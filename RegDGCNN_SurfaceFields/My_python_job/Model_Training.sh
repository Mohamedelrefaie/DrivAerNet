python run_pipeline.py \
    --stages train \
    --exp_name "Train_Test" \
    --dataset_path "$HOME/ML_Turbulent/DrivAerNet/RegDGCNN_SurfaceFields/My_python_job/Pressure_VTK" \
    --subset_dir "$HOME/ML_Turbulent/DrivAerNet/train_val_test_splits" \
    --cache_dir "$HOME/ML_Turbulent/DrivAerNet/RegDGCNN_SurfaceFields/My_python_job/Cache_data" \
    --num_points 10000 \
    --num_workers 1 \
    --batch_size 2 \
    --epochs 10 \
    --gpus "0"


