#!/bin/bash

export PYTHONPATH="./deps/wam"

new=$1
imagenet=$2
pattern=$3

if [ "$imagenet" = "true" ]; then
    imagenet_str="_imagenet"
    model_id="256x256_diffusion"
else
    imagenet_str=""
    model_id="Manojb/stable-diffusion-2-1-base"
fi

if [ "$new" = "false" ]; then
    script_name="run_tree_ring_watermark${imagenet_str}.py"
    fid_script_name="run_tree_ring_watermark${imagenet_str}_fid.py"

    python $script_name --run_name imgnet_no_attack --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 1000 --with_tracking --reference_model dummy
    python $script_name --run_name imgnet_rotation --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 1000 --r_degree 75 --with_tracking
    python $script_name --run_name imgnet_jpeg --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 1000 --jpeg_ratio 25 --with_tracking
    python $script_name --run_name imgnet_cropping --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 1000 --crop_scale 0.75 --crop_ratio 0.75 --with_tracking
    python $script_name --run_name imgnet_blurring --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 1000 --gaussian_blur_r 4 --with_tracking 
    python $script_name --run_name imgnet_noise --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 1000 --gaussian_std 0.1 --with_tracking
    python $script_name --run_name imgnet_color_jitter --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 1000 --brightness_factor 6 --with_tracking

    python $fid_script_name --run_name imgnet_fid_run --gt_data imagenet --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 1000 --with_tracking
else
    script_name="run_tree_ring_watermark${imagenet_str}_new.py"
    fid_script_name="run_tree_ring_watermark${imagenet_str}_fid_new.py"

    type=$4

    # 根据type参数设置synctype和syncpath
    if [ "$type" = "wam" ]; then
        synctype="wam"
        syncpath="./checkpoints/wam_mit.pth"
    elif [ "$type" = "sync_seal" ]; then
        synctype="sync_seal"
        syncpath="./checkpoints/syncmodel.jit.pt"
    else
        echo "Invalid type parameter. Must be 'wam' or 'sync_seal'"
        exit 1
    fi

    # 所有命令，使用传递的参数
    python $script_name --run_name imgnet_no_attack --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 1000 --with_tracking --reference_model dummy --synctype $synctype --syncpath $syncpath
    python $script_name --run_name imgnet_rotation --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 1000 --r_degree 75 --with_tracking --synctype $synctype --syncpath $syncpath
    python $script_name --run_name imgnet_jpeg --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 1000 --jpeg_ratio 25 --with_tracking --synctype $synctype --syncpath $syncpath
    python $script_name --run_name imgnet_cropping --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 1000 --crop_scale 0.75 --crop_ratio 0.75 --with_tracking --synctype $synctype --syncpath $syncpath
    python $script_name --run_name imgnet_blurring --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 1000 --gaussian_blur_r 4 --with_tracking --synctype $synctype --syncpath $syncpath
    python $script_name --run_name imgnet_noise --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 1000 --gaussian_std 0.1 --with_tracking --synctype $synctype --syncpath $syncpath
    python $script_name --run_name imgnet_color_jitter --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 1000 --brightness_factor 6 --with_tracking --synctype $synctype --syncpath $syncpath

    # FID命令
    python $fid_script_name --run_name imgnet_fid_run --gt_data imagenet --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 1000 --with_tracking --synctype $synctype --syncpath $syncpath
fi
