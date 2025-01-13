import os
import yaml
import math
import argparse

import lpips

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import models
import utils

from utils import PerceptualLoss
from test import eval_score

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=8, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader

def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        epoch_start = sv_file['epoch'] + 1
        if config['mode'] == 'stage2':
            discriminator = models.make(sv_file['discriminator'], load_sd=True).cuda()
            m_optimizer = utils.make_optimizer(model.parameters(), sv_file['m_optimizer'], load_sd=True)
            d_optimizer = utils.make_optimizer(discriminator.parameters(), sv_file['d_optimizer'], load_sd=True)
            percept_fn = PerceptualLoss()
            
            return model, discriminator, m_optimizer, d_optimizer, epoch_start, percept_fn
        else:
            optimizer = utils.make_optimizer(model.parameters(), sv_file['optimizer'], load_sd=True)
            if config.get('multi_step_lr') is None:
                lr_scheduler = None
            else:
                lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'], last_epoch=epoch_start)
            return model, optimizer, epoch_start, lr_scheduler
    else:
        epoch_start = 1
        if config['mode'] == 'stage2':
            sv_file = torch.load(config['pretrained'])
            sv_file['model']['name'] = config['model']['name']
            sv_file['model']['args'] = config['model']['args']
            model = models.make(sv_file['model'], load_sd=True).cuda() 
            discriminator = models.make(config['discriminator']).cuda()
            m_optimizer = utils.make_optimizer(model.parameters(), config['m_optimizer'])
            d_optimizer = utils.make_optimizer(discriminator.parameters(), config['d_optimizer'])
            percept_fn = PerceptualLoss().cuda()
            return model, discriminator, m_optimizer, d_optimizer, epoch_start, percept_fn
        else:
            model = models.make(config['model']).cuda()
            optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
            if config.get('multi_step_lr') is None:
                lr_scheduler = None
            else:
                lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

            log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
            return model, optimizer, epoch_start, lr_scheduler


def train_stage1(train_loader, model, optimizer):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    for batch in tqdm(train_loader, leave=False, desc='train'):        
        for k, v in batch.items():
            v = v.squeeze(0)
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        ref = (batch['ref'] - inp_sub) / inp_div
        gt = (batch['gt'] - gt_sub) / gt_div

        pred = model(inp, ref, batch['coord'], batch['cell'])
        loss = loss_fn(pred, gt)

        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None; loss = None

    return train_loss.item()

def train_stage2(train_loader, model, discriminator, m_optimizer, d_optimizer, percept_fn):
    model = model.cuda()
    discriminator = discriminator.cuda()
    percept_fn = percept_fn.cuda()
    
    model.train()
    discriminator.train()
    loss_fn = nn.L1Loss()
    adv_fn = nn.BCEWithLogitsLoss()
    
    m_train_loss = utils.Averager()
    m_rec_loss = utils.Averager()
    m_percept_loss = utils.Averager()
    m_fake_loss = utils.Averager()
    m_real_loss = utils.Averager()
    m_diff_loss = utils.Averager()

    d_train_loss = utils.Averager()
    d_fake_loss = utils.Averager()
    d_real_loss = utils.Averager()

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    for batch in tqdm(train_loader, leave=False, desc='train'):        
        for k, v in batch.items():
            v = v.squeeze(0)
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        ref = (batch['ref'] - inp_sub) / inp_div
        gt = (batch['gt'] - gt_sub) / gt_div

        lbl_shape = (inp.shape[0], 1, inp.shape[2], inp.shape[3])

        real_lbl = torch.ones(lbl_shape, dtype=inp.dtype, device=inp.device)
        fake_lbl = torch.zeros(lbl_shape, dtype=inp.dtype, device=inp.device)

        # model update
        pred, l_diff = model(inp, ref, batch['coord'], batch['cell'])
        l_diff = l_diff.mean() * 5e-2 # diffuision loss
        l_rec = loss_fn(pred, gt)  # l1 loss

        pred = pred*inp_div + inp_sub
        gt = gt*gt_div + gt_sub

        ih, iw = inp.shape[-2:]
        s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
        shape = [inp.shape[0], round(ih * s), round(iw * s), 3]
        pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
        gt = gt.view(*shape).permute(0, 3, 1, 2).contiguous()

        pred = pred.clamp(min=0.0, max=1.0)
        
        fake_out = discriminator(pred - gt.mean()) 
        real_out = discriminator(gt - pred.mean()) # hinder discriminator to become overfitted to GT Image; better results than GT only
        l_fake = adv_fn(fake_out, real_lbl) * 5e-2 # gan loss 1
        l_real = adv_fn(real_out, fake_lbl) * 5e-2 # gan loss 2
        
        l_percept = percept_fn(pred, gt).mean() * 0.1 # perceptual loss

        m_loss = l_rec + l_diff + l_fake + l_real + l_percept 

        m_train_loss.add(m_loss.item())
        m_rec_loss.add(l_rec.item())
        m_diff_loss.add(l_diff.item())
        m_fake_loss.add(l_fake.item())
        m_real_loss.add(l_real.item())
        m_percept_loss.add(l_percept.item())

        m_optimizer.zero_grad()
        m_loss.backward()
        m_optimizer.step()

        # discriminator update
        discriminator.zero_grad()
        
        pred = pred.detach()
        fake_out = discriminator(pred - gt.mean())
        real_out = discriminator(gt - pred.mean())
        
        d_fake = adv_fn(fake_out, fake_lbl)
        d_real = adv_fn(real_out, real_lbl)

        d_loss =  d_fake + d_real 
        
        d_train_loss.add(d_loss.item())
        d_real_loss.add(d_real.item())
        d_fake_loss.add(d_fake.item())

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
    return m_train_loss.item(), m_rec_loss.item(), m_diff_loss.item(), m_real_loss.item(), m_fake_loss.item(), m_percept_loss.item(), d_train_loss.item(), d_real_loss.item(), d_fake_loss.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    if config['mode'] == 'stage2':
        model, discriminator, m_optimizer, d_optimizer, epoch_start, percept_fn = prepare_training()
        lpips_fn = lpips.LPIPS(net='alex').cuda()
    else:
        model, optimizer, epoch_start, lr_scheduler = prepare_training()


    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    print("n_gpus : ", n_gpus)
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)
        if config['mode'] == 'stage2':
            discriminator = nn.parallel.DataParallel(discriminator)
            percept_fn = nn.parallel.DataParallel(percept_fn)
            lpips_fn = nn.parallel.DataParallel(lpips_fn)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        if config['mode'] == 'stage2':
            m_loss, rec_loss, diff_loss, real_loss, fake_loss, percept_loss, d_loss, d_real_loss, d_fake_loss = train_stage2(train_loader, model, discriminator, m_optimizer, d_optimizer, percept_fn)

            log_info.append('train: model loss={:.4f}, discriminator loss={:.4f}'.format(m_loss, d_loss))
            writer.add_scalars('model loss', {'total': m_loss, 'l1': rec_loss, 'diffusion': diff_loss, 'fake': fake_loss, 'real': real_loss, 'perceptual': percept_loss}, epoch)
            writer.add_scalars('discriminator loss', {'total': d_loss, 'real': d_real_loss, 'fake': d_fake_loss}, epoch)

            if n_gpus > 1:
                model_ = model.module
                discriminator_ = discriminator.module
            else:
                model_ = model
                discriminator_ = discriminator

            model_spec = config['model']
            discriminator_spec = config['discriminator']
            model_spec['sd'] = model_.state_dict()
            discriminator_spec['sd'] = discriminator_.state_dict()
            
            m_optimizer_spec = config['m_optimizer']
            d_optimizer_spec = config['d_optimizer']
            m_optimizer_spec['sd'] = m_optimizer.state_dict()
            d_optimizer_spec['sd'] = d_optimizer.state_dict()
            
            sv_file = {
                'model': model_spec,
                'discriminator': discriminator_spec,
                'm_optimizer': m_optimizer_spec,
                'd_optimizer': d_optimizer_spec,
                'epoch': epoch,
                'mode': 'stage2'
            }

        else:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            train_loss = train_stage1(train_loader, model, optimizer)
            if lr_scheduler is not None:
                lr_scheduler.step()

            log_info.append('train: loss={:.4f}'.format(train_loss))
            writer.add_scalars('loss', {'train': train_loss}, epoch)

            if n_gpus > 1:
                model_ = model.module
            else:
                model_ = model

            model_spec = config['model']
            model_spec['sd'] = model_.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()
            sv_file = {
                'model': model_spec,
                'optimizer': optimizer_spec,
                'epoch': epoch,
                'mode': 'stage1'
            }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
                if config['mode'] == 'stage2':
                    lpips_fn_ = lpips_fn.module
            else:
                model_ = model
                if config['mode'] == 'stage2':
                    lpips_fn_ = lpips_fn

            if config['mode'] == 'stage2':
                val_res, val_lpips, gt, pred = eval_score(val_loader, model_,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'),
                lpips_fn=lpips_fn_)
            else:
                val_res, *_ = eval_score(val_loader, model_,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'))

            if config['mode'] == 'stage2':
                log_info.append('val: psnr={:.4f}, lpips={:.4f}'.format(val_res, val_lpips))
                writer.add_scalars('psnr', {'val': val_res}, epoch)
                writer.add_scalars('lpips', {'val': val_lpips}, epoch)
                writer.add_images(tag='images/gt', img_tensor=gt, global_step=epoch)
                writer.add_images(tag='images/upsample', img_tensor=pred, global_step=epoch)
            else:
                log_info.append('val: psnr={:.4f}'.format(val_res))
                writer.add_scalars('psnr', {'val': val_res}, epoch)

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
