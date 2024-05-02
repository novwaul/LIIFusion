import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

import lpips

def batched_predict(model, inp, ref, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp, ref)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred

def eval_score(loader, model, data_norm=None, eval_type=None, eval_bsize=None, verbose=False, lpips_fn=None):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()
    val_lpips = utils.Averager()

    keep_pred = None
    keep_gt = None

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            v = v.squeeze(0)
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        ref = (batch['ref'] - inp_sub) / inp_div

        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, ref, batch['coord'], batch['cell'])
        else:
            pred = batched_predict(model, inp, ref,
                batch['coord'], batch['cell'], eval_bsize)

        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None or lpips_fn is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])
        
        if lpips_fn is not None:
            pred = (pred - 0.5) / 0.5
            l_pred = pred.clamp(-1, 1)
            gt = (batch['gt'] - 0.5) / 0.5
            gt = gt.clamp(-1, 1)
            lpips_score = lpips_fn(l_pred, gt)
            val_lpips.add(lpips_score.mean().item(), inp.shape[0])

        if keep_gt == None:
            keep_gt = batch['gt']
        if keep_pred == None:
            keep_pred = pred

    if lpips_fn is not None:
        if verbose:
            pbar.set_description('val psnr {:.4f}, lpips {:.4f}'.format(val_res.item(), val_lpips.item()))
        return val_res.item(), val_lpips.item(), keep_gt, keep_pred
    else:
        if verbose:
            pbar.set_description('val psnr {:.4f}'.format(val_res.item()))
        return val_res.item(), keep_gt, keep_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    train_config = torch.load(args.model)
    model_spec = train_config['model']
    model = models.make(model_spec, load_sd=True).cuda()

    if train_config['mode'] == 'stage2':
        lpips_fn = lpips.LPIPS(net='alex').cuda()
    else:
        lpips_fn = None

    result = eval_score(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True, 
        lpips_fn=lpips_fn)
    
    if train_config['mode'] == 'stage2':
        res, lpips_score, *_ = result
        print('result psnr: {:.4f}, lpips: {:.4f}'.format(res, lpips_score))
    else:
        res, *_ = result
        print('result psnr: {:.4f}'.format(res))