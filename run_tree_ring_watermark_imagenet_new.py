import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics
from torchvision.transforms.functional import to_tensor, to_pil_image
from tree_ring_watermark.sync.factory import get_sync_model

import torch

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from optim_utils import *
from io_utils import *


def main(args):
    table = None
    if args.with_tracking:
        wandb.init(project='diffusion_watermark', name=args.run_name, tags=['latent_watermark_fourier_openai'])
        wandb.config.update(args)
        table = wandb.Table(columns=['gen_no_w', 'gen_w', 'no_w_metric', 'w_metric'])

    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    args.timestep_respacing = f"ddim{args.num_inference_steps}"

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    shape = (args.num_images, 3, args.image_size, args.image_size)

    # ground-truth patch
    gt_patch = get_watermarking_pattern(None, args, device, shape)

    results = []
    no_w_metrics = []
    w_metrics = []

    save_dir = f"save/imagenet/{args.run_name}_{args.w_pattern}_new"

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed
        
        ### generation
        model_kwargs = {}
        if args.class_cond:
            classes = torch.randint(
                low=0, high=NUM_CLASSES, size=(args.num_images,), device=device
            )
            model_kwargs["y"] = classes
            
        # generation without watermarking
        set_random_seed(seed)
        init_latents_no_w = torch.randn(*shape, device=device)
        outputs_no_w = diffusion.ddim_sample_loop(
                    model=model,
                    shape=shape,
                    noise=init_latents_no_w,
                    model_kwargs=model_kwargs,
                    device=device,
                    return_image=True,
                )
        orig_image_no_w = outputs_no_w[0]
        
        # generation with watermarking
        if init_latents_no_w is None:
            set_random_seed(seed)
            init_latents_w = torch.randn(*shape, device=device)
        else:
            init_latents_w = copy.deepcopy(init_latents_no_w)

        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)

        # inject watermark
        init_latents_w = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args)

        outputs_w = diffusion.ddim_sample_loop(
                    model=model,
                    shape=shape,
                    noise=init_latents_w,
                    model_kwargs=model_kwargs,
                    device=device,
                    return_image=True,
                )
        orig_image_w = outputs_w[0]

        ### test watermark
        # embedding
        synctype = args.synctype
        syncpath = args.syncpath
        sync_model = get_sync_model(synctype, syncpath, device)
        # scripted = torch.jit.load("syncmodel.jit.pt").to(device).eval()
        orig_tensor_w = to_tensor(orig_image_w).unsqueeze(0).to(device)
        with torch.no_grad():
            embedded_image = sync_model.add_sync(2.0 * orig_tensor_w - 1.0)
            embedded_image = (embedded_image + 1.0) / 2.0
        orig_image_w_emb = to_pil_image(embedded_image.squeeze().cpu())

        # distortion
        orig_image_no_w_auged, orig_image_w_auged_emb = image_distortion(orig_image_no_w, orig_image_w_emb, seed, args)
        
        # synchronization
        orig_tensor_no_w_auged = to_tensor(orig_image_no_w_auged).unsqueeze(0).to(device)
        with torch.no_grad():
            orig_tensor_no_w_auged_sync = sync_model.remove_sync(2.0 * orig_tensor_no_w_auged - 1.0)
            orig_tensor_no_w_auged_sync = (orig_tensor_no_w_auged_sync + 1.0) / 2.0
            # orig_tensor_no_w_auged_sync = orig_tensor_no_w_auged
        # pred_pts_no_w = det_no_w["preds_pts"]
        # orig_tensor_no_w_auged_sync = scripted.unwarp(orig_tensor_no_w_auged, pred_pts_no_w, original_size=orig_tensor_no_w_auged.shape[-2:])
        orig_image_no_w_auged_sync = to_pil_image(orig_tensor_no_w_auged_sync.squeeze().cpu())

        orig_tensor_w_auged_emb = to_tensor(orig_image_w_auged_emb).unsqueeze(0).to(device)
        with torch.no_grad():
            orig_tensor_w_auged_sync = sync_model.remove_sync(2.0 * orig_tensor_w_auged_emb - 1.0)
            orig_tensor_w_auged_sync = (orig_tensor_w_auged_sync + 1.0) / 2.0
        # pred_pts_w = det_w["preds_pts"]
        # orig_tensor_w_auged_sync = scripted.unwarp(orig_tensor_w_auged_emb, pred_pts_w, original_size=orig_tensor_w_auged_emb.shape[-2:])
        orig_image_w_auged_sync = to_pil_image(orig_tensor_w_auged_sync.squeeze().cpu())

        # save images
        if i < 50:
            os.makedirs(f"{save_dir}/{i}", exist_ok=True)
            orig_image_no_w.save(f"{save_dir}/{i}/orig_no_w.png")
            orig_image_w.save(f"{save_dir}/{i}/orig_w.png")
            orig_image_no_w_auged.save(f"{save_dir}/{i}/orig_no_w_auged.png")
            orig_image_no_w_auged_sync.save(f"{save_dir}/{i}/orig_no_w_auged_sync.png")
            orig_image_w_emb.save(f"{save_dir}/{i}/orig_w_emb.png")
            orig_image_w_auged_emb.save(f"{save_dir}/{i}/orig_w_auged_emb.png")
            orig_image_w_auged_sync.save(f"{save_dir}/{i}/orig_w_auged_sync.png")

        # reverse img without watermarking
        reversed_latents_no_w = diffusion.ddim_reverse_sample_loop(
                model=model,
                shape=shape,
                image=orig_image_no_w_auged_sync,
                model_kwargs=model_kwargs,
                device=device,
            )

        # reverse img with watermarking
        reversed_latents_w = diffusion.ddim_reverse_sample_loop(
                model=model,
                shape=shape,
                image=orig_image_w_auged_sync,
                model_kwargs=model_kwargs,
                device=device,
            )

        # eval
        no_w_metric, w_metric = eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args)

        # TODO 下面评测部分尚未修改
        results.append({
            'no_w_metric': no_w_metric, 'w_metric': w_metric,
        })

        no_w_metrics.append(-no_w_metric)
        w_metrics.append(-w_metric)

        if args.with_tracking:
            if (args.reference_model is not None) and (i < args.max_num_log_image):
                # log images when we use reference_model
                table.add_data(wandb.Image(orig_image_no_w), wandb.Image(orig_image_w_emb), no_w_metric, w_metric)
            else:
                table.add_data(None, None, no_w_metric, w_metric)

    # roc
    preds = no_w_metrics +  w_metrics
    t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)

    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr<.01)[0][-1]]

    if args.with_tracking:
        wandb.log({'Table': table})
        wandb.log({'auc': auc, 'acc':acc, 'TPR@1%FPR': low})
        
    print(f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low}')
    print(mean(no_w_metrics), mean(w_metrics))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='256x256_diffusion')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)

    # watermark
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)

    # sync model
    parser.add_argument('--synctype', default='wam', type=str, help='sync model type')
    parser.add_argument('--syncpath', required=True, type=str, help='path to sync model')

    # for image distortion
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)

    args = parser.parse_args()

    args.__dict__.update(model_and_diffusion_defaults())
    args.__dict__.update(read_json(f'{args.model_id}.json'))

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)