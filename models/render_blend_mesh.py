import torch
from torch import searchsorted
import numpy as np
# from torchsearchsorted import searchsorted
from manopth.manolayer import ManoLayer
from utils.deformation import warp_observation_to_canonical

__all__ = ['render_rays_blend']

shapes = torch.zeros(1, 10)

"""
Function dependencies: (-> means function calls)

@render_rays -> @inference

@render_rays -> @sample_pdf if there is fine model
"""

def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = searchsorted(cdf, u, side='right')
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


def inference(model, embedding_xyz, embedding_dir, xyz_, dir_embedded, N_rays, chunk, \
              weights_only=False, poses=None, mano_layer=None, global_translation=None):
        """
        Helper function that performs model inference.

        Inputs:
            model: NeRF model (coarse or fine)
            embedding_xyz: embedding module for xyz
            embedding_dir: embedding module for direction
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_: (N_rays, 3) ray directions
            dir_embedded: (N_rays, embed_dir_channels) embedded directions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            weights_only: do inference on sigma only or not

        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz_.shape[1]
        # Embed directions
        xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)

        if global_translation is not None:
            xyz_ -= global_translation

        if poses is not None:
            vertices, _, joint_trans_mat = mano_layer(poses, shapes)
            vertices = vertices[0]
            joint_trans_mat = joint_trans_mat[0]
            joint_trans_mat = joint_trans_mat.permute(2, 0, 1)
            xyz_ /= 12
            trans_mat, signed_dis = warp_observation_to_canonical(xyz_, vertices, mano_layer.th_faces, \
                                                                  joint_trans_mat, mano_layer.th_weights, return_signed_dis=True)
            trans_mat = torch.inverse(trans_mat)
            xyz_ = torch.cat([xyz_, torch.ones(xyz_.shape[0], 1, device=xyz_.device)], dim=1)
            xyz_ = xyz_.unsqueeze(-1)
            xyz_ = torch.bmm(trans_mat.cuda(), xyz_)
            # xyz_ = xyz_[:, :3] / xyz_[:, 3:]
            # xyz_ = xyz_.squeeze()
            xyz_ = xyz_[:, :3].squeeze()
            xyz_ *= 12
            
        if not weights_only:
            if poses is None:
                dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                        # (N_rays*N_samples_, embed_dir_channels)
            else:
                xyz_ = xyz_.view(-1, N_samples_, 3)
                dirs = xyz_[:, 1:] - xyz_[:, :-1]
                xyz_ = xyz_.view(-1, 3)
                dirs = torch.cat([dirs, dirs[:, -1:]], dim=1)
                assert torch.isnan(dirs).any() == False, "dirs has nan"
                dir_norm = torch.linalg.norm(dirs, dim=-1, keepdim=True)
                dir_norm.data = torch.where(dir_norm.data == 0, torch.ones_like(dir_norm.data), dir_norm.data)
                # dir_norm.data = torch.where(dir_norm.data == 0, torch.zeros_like(dir_norm.data) + 1e-8, dir_norm.data)
                # dir_norm = dir_norm + 1e-8
                dirs = dirs / dir_norm
                assert torch.isnan(dirs).any() == False, "dirs has nan after norm"
                dirs = dirs.view(-1, 3)
                dir_embedded = embedding_dir(dirs)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, chunk):
            # Embed positions by chunk
            xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
            if not weights_only:
                xyzdir_embedded = torch.cat([xyz_embedded,
                                             dir_embedded[i:i+chunk]], 1)
            else:
                xyzdir_embedded = xyz_embedded
            out_chunks += [model(xyzdir_embedded, sigma_only=weights_only)]

        out = torch.cat(out_chunks, 0)
        result = {}
        if poses is not None:
            result['signed_dis'] = signed_dis.view(N_rays, N_samples_)

        if weights_only:
            sigmas = out.view(N_rays, N_samples_)
            result['sigma'] = sigmas
        else:
            rgbsigma = out.view(N_rays, N_samples_, 4)
            rgbs = rgbsigma[..., :3] # (N_rays, N_samples_, 3)
            sigmas = rgbsigma[..., 3] # (N_rays, N_samples_)
            result['sigma'] = sigmas
            result['rgb'] = rgbs

        return result

def volume_rendering(z_vals, dir_, sigmas, rgbs, noise_std=1, white_back=False, weights_only=False):
    # Convert these values using volume rendering (Section 4)
    deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

    noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

    # compute alpha by the formula (3)
    # alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)
    alphas = 1-torch.exp(-deltas*torch.nn.Softplus()(sigmas+noise)) # (N_rays, N_samples_)
    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
    weights = \
        alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
    weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
    if weights_only:
        return weights

    # compute final weighted outputs
    rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
    depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

    if white_back:
        rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

    return rgb_final, depth_final, weights

def render_rays_blend(hand_models,
                hand_embeddings,
                object_models,
                object_embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False,
                poses=None,
                mano_layer=None,
                global_translation=None
                ):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """
    # Extract models from lists
    hand_model_coarse = hand_models[0]
    hand_embedding_xyz = hand_embeddings[0]
    hand_embedding_dir = hand_embeddings[1]
    object_model_coarse = object_models[0]
    object_embedding_xyz = object_embeddings[0]
    object_embedding_dir = object_embeddings[1]

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)

    # Embed direction
    hand_dir_embedded = hand_embedding_dir(rays_d) # (N_rays, embed_dir_channels)
    object_dir_embedded = object_embedding_dir(rays_d) # (N_rays, embed_dir_channels)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)
    
    xyz_coarse_sampled_copy = xyz_coarse_sampled.clone()

    if test_time:
        hand_result_coarse = \
            inference(hand_model_coarse, hand_embedding_xyz, hand_embedding_dir, xyz_coarse_sampled, \
                      hand_dir_embedded, N_rays=N_rays, chunk=chunk, \
                      weights_only=True, poses=poses, mano_layer=mano_layer, global_translation=global_translation)
        object_result_coarse = \
            inference(object_model_coarse, object_embedding_xyz, object_embedding_dir, xyz_coarse_sampled_copy, \
                      object_dir_embedded, N_rays=N_rays, chunk=chunk, \
                      weights_only=True)
    else:
        hand_result_coarse = \
            inference(hand_model_coarse, hand_embedding_xyz, hand_embedding_dir, xyz_coarse_sampled, \
                      hand_dir_embedded, N_rays=N_rays, chunk=chunk, \
                      weights_only=False, poses=poses, mano_layer=mano_layer, global_translation=global_translation)
        object_result_coarse = \
            inference(object_model_coarse, object_embedding_xyz, object_embedding_dir, xyz_coarse_sampled_copy, \
                      object_dir_embedded, N_rays=N_rays, chunk=chunk, \
                      weights_only=False)

    # blend hand and object coarse results to get weights for fine model
    hand_sigma = hand_result_coarse['sigma']
    object_sigma = object_result_coarse['sigma']
    mask_coarse = (hand_result_coarse['signed_dis'] <= 0.0).to(rays.device)

    result = {}

    sigma_coarse = torch.where(mask_coarse, hand_sigma, object_sigma)
    if test_time:
        weights_coarse = volume_rendering(z_vals, rays_d, sigma_coarse, None, noise_std=noise_std, white_back=white_back, weights_only=True)
        result['opacity_coarse'] = weights_coarse.sum(1)
    else:
        rgbs_coarse = torch.where(mask_coarse.unsqueeze(-1), hand_result_coarse['rgb'], object_result_coarse['rgb'])
        rgb_coarse, depth_coarse, weight_coarse = volume_rendering(z_vals, rays_d, sigma_coarse, rgbs_coarse, \
                                                                   noise_std=noise_std, white_back=white_back, weights_only=False)
        result['rgb_coarse'] = rgb_coarse
        result['depth_coarse'] = depth_coarse
        result['opacity_coarse'] = weight_coarse.sum(1)

    if N_importance > 0: # sample points for fine model
        # weights_coarse = hand_result['sigma']
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb==0)).detach() # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
                           # (N_rays, N_samples+N_importance, 3)

        xyz_fine_sampled_copy = xyz_fine_sampled.clone()

        hand_model_fine = hand_models[1]
        hand_result_fine = \
            inference(hand_model_fine, hand_embedding_xyz, hand_embedding_dir, xyz_fine_sampled, \
                      hand_dir_embedded, N_rays=N_rays, chunk=chunk, \
                      weights_only=False, poses=poses, mano_layer=mano_layer, global_translation=global_translation)
        
        object_model_fine = object_models[1]
        object_result_fine = \
            inference(object_model_fine, object_embedding_xyz, object_embedding_dir, xyz_fine_sampled_copy, \
                      object_dir_embedded, N_rays=N_rays, chunk=chunk, \
                      weights_only=False)
        
        # blend hand and object fine results
        hand_sigma = hand_result_fine['sigma']
        object_sigma = object_result_fine['sigma']
        hand_rgbs = hand_result_fine['rgb']
        object_rgbs = object_result_fine['rgb']
        mask_fine = (hand_result_fine['signed_dis'] <= 0.0).to(rays.device)

        sigma_fine = torch.where(mask_fine, hand_sigma, object_sigma)
        rgbs_fine = torch.where(mask_fine.unsqueeze(-1), hand_rgbs, object_rgbs)
        rgb_fine, depth_fine, weight_fine = volume_rendering(z_vals, rays_d, sigma_fine, rgbs_fine, noise_std=noise_std, white_back=white_back, weights_only=False)

    result['rgb_fine'] = rgb_fine
    result['depth_fine'] = depth_fine
    result['opacity_fine'] = weight_fine.sum(1)
    return result
