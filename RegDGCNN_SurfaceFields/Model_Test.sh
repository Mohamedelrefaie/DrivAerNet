python test_single_vtk.py \
    --model_checkpoint "experiments/DrivAerNet_Pressure/best_model.pth" \
    --vtk_file "$HOME/ML_Turbulent/Data_Pressure_Field/Data_Pressure/PressureVTK/N_S_WWS_WM/N_S_WWS_WM_001.vtk" \
    --output_dir "visualizations" \
    --num_points 10000 \
    --k 40 \
    --emb_dims 1024 \
    --dropout 0.4

