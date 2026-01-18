#!/bin/bash

export PYTHONPATH="./deps/wam"

new=$1
imagenet=$2
pattern=$3

if [ "$imagenet" = "true" ]; then
    imagenet_str="_imagenet"
    model_id="256x256_diffusion"
    run_prefix="imgnet_"
else
    imagenet_str=""
    model_id="Manojb/stable-diffusion-2-1-base"
    run_prefix="stable_"
fi

if [ "$new" = "false" ]; then
    script_name="run_tree_ring_watermark${imagenet_str}.py"
    fid_script_name="run_tree_ring_watermark${imagenet_str}_fid.py"

    if [ "$imagenet" = "true" ]; then
        python $script_name --run_name ${run_prefix}no_attack --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 100 --with_tracking --reference_model dummy
    else
        python $script_name --run_name ${run_prefix}no_attack --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 100 --with_tracking --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k
    fi
    python $script_name --run_name ${run_prefix}rotation --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 100 --r_degree 75 --with_tracking
    
   python $script_name --run_name ${run_prefix}jpeg --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 100 --jpeg_ratio 25 --with_tracking
    
    python $script_name --run_name ${run_prefix}cropping --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 100 --crop_scale 0.75 --crop_ratio 0.75 --with_tracking
    
    python $script_name --run_name ${run_prefix}blurring --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 100 --gaussian_blur_r 4 --with_tracking
    
    python $script_name --run_name ${run_prefix}noise --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 100 --gaussian_std 0.1 --with_tracking
    
    python $script_name --run_name ${run_prefix}color_jitter --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 100 --gaussian_std 0.1 --with_tracking
    
    python $script_name --run_name ${run_prefix}color_jitter --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 100 --brightness_factor 6 --with_tracking

    # python $fid_script_name --run_name ${run_prefix}fid_run --gt_data imagenet --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 100 --with_tracking
    
    echo "所有任务执行完成！"

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
    if [ "$imagenet" = "true" ]; then
        python $script_name --run_name ${run_prefix}no_attack --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 100 --with_tracking --reference_model dummy --synctype $synctype --syncpath $syncpath
    else
        python $script_name --run_name ${run_prefix}no_attack --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 100 --with_tracking --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k --synctype $synctype --syncpath $syncpath
    fi
    
    python $script_name --run_name ${run_prefix}rotation --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 100 --r_degree 75 --with_tracking --synctype $synctype --syncpath $syncpath
    
    python $script_name --run_name ${run_prefix}jpeg --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 100 --jpeg_ratio 25 --with_tracking --synctype $synctype --syncpath $syncpath
    
    python $script_name --run_name ${run_prefix}cropping --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 100 --crop_scale 0.75 --crop_ratio 0.75 --with_tracking --synctype $synctype --syncpath $syncpath
    
    python $script_name --run_name ${run_prefix}blurring --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 100 --gaussian_blur_r 4 --with_tracking --synctype $synctype --syncpath $syncpath
    
    python $script_name --run_name ${run_prefix}noise --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 100 --gaussian_std 0.1 --with_tracking --synctype $synctype --syncpath $syncpath
    
    python $script_name --run_name ${run_prefix}color_jitter --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 100 --brightness_factor 6 --with_tracking --synctype $synctype --syncpath $syncpath

    # if [ "$imagenet" = "true" ]; then
    #     CUDA_VISIBLE_DEVICES=7 python $fid_script_name --run_name ${run_prefix}fid_run --gt_data imagenet --model_id $model_id --w_radius 10 --w_channel 2 --w_pattern $pattern --start 0 --end 100 --with_tracking --synctype $synctype --syncpath $syncpath #&
    #     pid8=$!
    # else
    #     CUDA_VISIBLE_DEVICES=7 python $fid_script_name --run_name ${run_prefix}fid_run --w_channel 3 --w_pattern $pattern --start 0 --end 100 --with_tracking --run_no_w --synctype $synctype --syncpath $syncpath #&
    #     pid8=$!
    # fi
# #python run_tree_ring_watermark_fid.py --run_name fid_run --w_channel 3 --w_pattern ring --start 0 --end 5000 --with_tracking --run_no_w
    
    echo "所有任务执行完成！"
fi
