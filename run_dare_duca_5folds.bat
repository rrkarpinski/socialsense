@echo off

REM --- First command: duca ---
for /L %%i in (0,1,4) do (
    echo Running duca, fold %%i
    cd D:\projects\DUCA
    python main.py --experiment_id officemanners_b60_dil_fold%%i --seed 42 --model duca --dataset custom-hdf5-regression --buffer_size 60 --aux shape --lr 0.05 --n_epochs 50 --img_size 64 --shape_filter sobel --batch_size 8 --minibatch_size 8 --output_dir ./output/ --loss_wt 0.1 0.1 0.01 0.01 --ema_alpha 0.999 --ema_update_freq 0.06 --loss_type l2 --save_model --alpha_mm 0.1 0.1 --beta_mm 0.1 0.1 > logs_duca_fold%%i.txt 2>&1
)

REM --- Second command: maxd_ema ---
for /L %%i in (0,1,4) do (
    echo Running maxd_ema, fold %%i
    cd D:\projects\DARE
    python main.py --experiment_id officemanners_b60_dil_fold%%i --model maxd_ema --dataset custom-hdf5-regression --img_size 64 --num_tasks 6 --alpha 0.1 --beta 0.2 --maximize_task hcr --maxd_weight 0.1 --mind_weight 1 --logitb_weight 1 --logitc_weight 1 --iterative_buffer --supcon_weight 0.0 --supcon_temp 0.8 --frozen_supcon --intermediate_sampling --std 4 --reduce_lr --each_epoch --buffer_size 60 --lr 0.04 --batch_size 16 --minibatch_size 16 --n_epochs 50 --output_folder ./output --tensorboard --plot_results > logs_maxd_fold%%i.txt 2>&1
)