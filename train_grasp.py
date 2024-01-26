import torch
from utils import *
from collections import defaultdict
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import json

from models.rendering import *
from models.nerf import *
from mano.manolayer import ManoLayer

import metrics

from datasets import dataset_dict
from datasets.llff import *

from models.render_blend_mesh import render_rays_blend

torch.backends.cudnn.benchmark = True

def load_dataset(path, kind='blender', dataset_type='train', img_wh=(200, 200)):
    dataset = dataset_dict[kind](path, dataset_type, img_wh=img_wh)
    return dataset

def load_models(hand_ckpt_path, object_ckpt_path):
    hand_embedding_xyz = Embedding(3, 10)
    hand_embedding_dir = Embedding(3, 4)
    object_embedding_xyz = Embedding(3, 10)
    object_embedding_dir = Embedding(3, 4)

    hand_nerf_coarse = NeRF()
    hand_nerf_fine = NeRF()
    object_nerf_coarse = NeRF()
    object_nerf_fine = NeRF()

    load_ckpt(hand_nerf_coarse, hand_ckpt_path, model_name='nerf_coarse')
    load_ckpt(hand_nerf_fine, hand_ckpt_path, model_name='nerf_fine')
    load_ckpt(object_nerf_coarse, object_ckpt_path, model_name='nerf_coarse')
    load_ckpt(object_nerf_fine, object_ckpt_path, model_name='nerf_fine')

    hand_nerf_coarse.cuda().eval()
    hand_nerf_fine.cuda().eval()
    object_nerf_coarse.cuda().eval()
    object_nerf_fine.cuda().eval()

    return [hand_nerf_coarse, hand_nerf_fine], [hand_embedding_xyz, hand_embedding_dir], \
        [object_nerf_coarse, object_nerf_fine], [object_embedding_xyz, object_embedding_dir]

@torch.no_grad()
def f_trans(hand_models, hand_embeddings, object_models, object_embeddings, \
            rays, white_back, N_samples = 64, N_importance = 64, use_disp=False, \
            chunk = 1024*32*4, poses = None, mano_layer = None, global_translation=None):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays_blend(hand_models,
                        hand_embeddings,
                        object_models,
                        object_embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        white_back,
                        test_time=True,
                        poses=poses,
                        mano_layer=mano_layer,
                        global_translation=global_translation)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results

def f_trans_grad(hand_models, hand_embeddings, object_models, object_embeddings, \
            rays, white_back, N_samples = 64, N_importance = 64, use_disp=False, \
            chunk = 1024*32*4, poses = None, mano_layer = None, global_translation=None):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays_blend(hand_models,
                        hand_embeddings,
                        object_models,
                        object_embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        white_back,
                        test_time=True,
                        poses=poses,
                        mano_layer=mano_layer,
                        global_translation=global_translation)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results

def train(poses, global_translation, epochs, rounds, batch_size, lr, img_wh, dataset, mano_layer, \
          hand_models, hand_embeddings, object_models, object_embeddings, N_samples, N_importance, use_disp, chunk):
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([poses, global_translation], lr=lr)
    length = img_wh[0] * img_wh[1]
    result_dict = {}
    sample_img_idx = 0
    img_idx = 0

    for epoch in range(epochs):
        tr = tqdm(range(rounds))
        tr.set_description(f'Epoch {epoch}')
        for round in tr:
            # img_idx = torch.randint(low=0, high=100, size=(1, ))
            if round % 50 == 0:
                img_idx = torch.randint(low=0, high=100, size=(1, ))
            # img_idx = sample_img_idx
            idx = torch.randint(low=0, high=length, size=(batch_size,)) + img_idx * length
            idx = idx.numpy()
            sample = dataset[idx]
            rays = sample['rays'].cuda()

            results = f_trans_grad(hand_models, hand_embeddings, object_models, object_embeddings, rays, \
                                   white_back=dataset.white_back, N_samples=N_samples, N_importance=N_importance, \
                                   use_disp=use_disp, chunk=chunk, poses=poses, mano_layer=mano_layer, global_translation=global_translation)

            optimizer.zero_grad()
            loss_val = loss(results['rgb_fine'], sample['rgbs'].cuda())
            loss_val.backward()
            tr.set_postfix({'loss': loss_val.item()})
            optimizer.step()
        
        result_dict[f'epoch_{epoch}'] = poses.tolist()

        with open('./results/grasp_cup.json', 'w') as f:
            f.write(json.dumps(result_dict, indent=2))

        show_compare_fig(dataset[sample_img_idx*length:(sample_img_idx+1)*length], poses, global_translation, mano_layer, img_wh, epoch, \
                         hand_models=hand_models, hand_embeddings=hand_embeddings, object_models=object_models, object_embeddings=object_embeddings, \
                         white_back=dataset.white_back, N_samples=N_samples, N_importance=N_importance, use_disp=use_disp, chunk=chunk)

def show_compare_fig(sample, poses, global_translation, mano_layer, img_wh, epoch, \
                     hand_models, hand_embeddings, object_models, object_embeddings, white_back, N_samples, N_importance, use_disp, chunk):
    rays = sample['rays'].cuda()
    results = f_trans(hand_models, hand_embeddings, object_models, object_embeddings, rays, \
                      white_back=white_back, N_samples=N_samples, N_importance=N_importance, \
                      use_disp=use_disp, chunk=chunk, poses=poses, mano_layer=mano_layer, global_translation=global_translation)
    torch.cuda.synchronize()
    img_gt = sample['rgbs'].view(img_wh[1], img_wh[0], 3)
    img_pred = results['rgb_fine'].view(img_wh[1], img_wh[0], 3).cpu().numpy()
    alpha_pred = results['opacity_fine'].view(img_wh[1], img_wh[0]).cpu().numpy()
    depth_pred = results['depth_fine'].view(img_wh[1], img_wh[0])

    plt.subplots(figsize=(15, 8))
    plt.tight_layout()
    plt.subplot(221)
    plt.title('GT')
    plt.imshow(img_gt)
    plt.subplot(222)
    plt.title('pred')
    plt.imshow(img_pred)
    plt.subplot(223)
    plt.title('depth')
    plt.imshow(visualize_depth(depth_pred).permute(1,2,0))
    plt.savefig(f'./results/grasp_cup_epoch_{epoch}.png')

def prepare():
    img_wh = (200, 200)
    dataset_path = './data/nerf_synthetic/grasp_cup/'
    dataset = load_dataset(path=dataset_path, img_wh=img_wh)

    hand_ckpt_path = './ckpts/hand_flat_same/epoch=7.ckpt'
    object_ckpt_path = './ckpts/cup_s8/epoch=7.ckpt'
    hand_models, hand_embeddings, object_models, object_embeddings = load_models(hand_ckpt_path, object_ckpt_path)

    N_samples = 64
    N_importance = 64
    use_disp = False
    chunk = 1024*32*4

    ncomps = 45
    poses = torch.zeros(1, ncomps + 3, dtype=torch.float32, requires_grad=True)
    # poses = torch.randn(1, ncomps + 3, dtype=torch.float32, requires_grad=True)
    # poses.data /= 100
    mano_layer = ManoLayer(mano_root='./mano/models', use_pca=False, ncomps=ncomps, flat_hand_mean=True)
    shapes = torch.zeros(1, 10)
    global_translations = torch.zeros(1, 3, dtype=torch.float32).cuda()
    global_translations.requires_grad = True

    epochs = 20
    rounds = 100
    batch_size = 1000
    lr=1e-3
    train(poses=poses, global_translation=global_translations, epochs=epochs, rounds=rounds, batch_size=batch_size, \
          lr=lr, img_wh=img_wh, dataset=dataset, mano_layer=mano_layer,\
          hand_models=hand_models, hand_embeddings=hand_embeddings, object_models=object_models, object_embeddings=object_embeddings, \
          N_samples=N_samples, N_importance=N_importance, use_disp=use_disp, chunk=chunk)
    
prepare()