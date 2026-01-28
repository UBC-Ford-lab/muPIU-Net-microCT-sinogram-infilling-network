# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.

Supports resuming from partial completion - skips tiles that already have output files.
"""

import os
import argparse
import torch as th
import torch.nn.functional as F
import time
import gc
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util
from pathlib import Path

# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass


from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)  # noqa: E402

def toU8(sample):
    """Convert from [-1, 1] float to uint8 [0, 255] - DEPRECATED, use toU16"""
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


def toU16(sample):
    """Convert from [-1, 1] float to uint16 [0, 65535] - preserves precision"""
    if sample is None:
        return sample

    sample = ((sample + 1) * 32767.5).clamp(0, 65535).to(th.int32)  # Use int32 to avoid overflow
    sample = sample.to(th.uint16)  # Convert to uint16
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


def main(conf: conf_mgt.Default_Conf, resume: bool = True):

    print("Start", conf['name'])
    if resume:
        print("Resume mode ENABLED - will skip tiles that already have output files")

    device = dist_util.dev(conf.get('device'))


    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress

    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(
                conf.classifier_path), map_location="cpu")
        )

        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    else:
        cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...")
    all_images = []

    dset = 'eval'

    eval_name = conf.get_default_eval_name()

    dl = conf.get_dataloader(dset=dset, dsName=eval_name)

    # Get output directory for resume check
    output_dir = None
    if resume:
        try:
            output_dir = Path(conf['data'][dset][eval_name]['paths']['srs'])
            if not output_dir.is_absolute():
                # Relative paths are relative to RePaint directory
                output_dir = Path.cwd() / output_dir
            print(f"Checking for existing outputs in: {output_dir}")
        except (KeyError, TypeError):
            print("WARNING: Could not determine output directory for resume check")
            resume = False

    # Track progress
    total_tiles = 0
    skipped_tiles = 0
    processed_tiles = 0
    batch_num = 0

    for batch in iter(dl):
        batch_num += 1
        img_names = batch['GT_name']
        batch_size_current = len(img_names)
        total_tiles += batch_size_current

        # Resume check: skip if ALL tiles in batch already have output files
        if resume and output_dir is not None:
            existing_count = 0
            for name in img_names:
                # Convert to .png extension if needed
                out_name = Path(name).stem + '.png'
                out_path = output_dir / out_name
                if out_path.exists():
                    existing_count += 1

            if existing_count == batch_size_current:
                # All tiles in this batch already exist - skip
                skipped_tiles += batch_size_current
                if batch_num % 10 == 0:
                    print(f"  Batch {batch_num}: Skipped (all {batch_size_current} tiles already exist)")
                continue
            elif existing_count > 0:
                # Partial batch - still need to process (will overwrite existing)
                print(f"  Batch {batch_num}: Processing ({existing_count}/{batch_size_current} already exist, will reprocess)")

        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}

        model_kwargs["gt"] = batch['GT']

        gt_keep_mask = batch.get('gt_keep_mask')
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        if conf.cond_y is not None:
            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * conf.cond_y
        else:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=device
            )
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )


        result = sample_fn(
            model_fn,
            (batch_size, 3, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf
        )
        # Use toU16 to preserve precision (uint16 instead of uint8)
        srs = toU16(result['sample'])
        gts = toU16(result['gt'])
        lrs = toU16(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
                    th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))

        gt_keep_masks = toU16((model_kwargs.get('gt_keep_mask') * 2 - 1))

        conf.eval_imswrite(
            srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
            img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False)

        processed_tiles += batch_size_current

        # Progress update every 10 batches
        if batch_num % 10 == 0:
            print(f"  Batch {batch_num}: Processed {processed_tiles} tiles, skipped {skipped_tiles} (already done)")

        # Periodic memory cleanup to prevent Bus errors on long HPC runs
        # Clear CUDA cache and run garbage collection every 50 batches
        if batch_num % 50 == 0:
            if th.cuda.is_available():
                th.cuda.empty_cache()
            gc.collect()

    print("")
    print("=" * 60)
    print("Sampling complete!")
    print(f"  Total tiles in range: {total_tiles}")
    print(f"  Skipped (already done): {skipped_tiles}")
    print(f"  Newly processed: {processed_tiles}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from existing output files (skip already completed tiles)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='Disable resume - reprocess all tiles even if output exists')
    args = parser.parse_args()

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.conf_path))
    main(conf_arg, resume=args.resume)
