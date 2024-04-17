import torch
from tqdm.auto import tqdm
import numpy as np


def init_from_pointe(prompt):
    from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
    from point_e.diffusion.sampler import PointCloudSampler
    from point_e.models.download import load_checkpoint
    from point_e.models.configs import MODEL_CONFIGS, model_from_config
    from point_e.util.plotting import plot_point_cloud

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('creating base model...')
    base_name = 'base40M-textvec'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
    print('downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))
    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))
    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
    )
    # Produce a sample from the model.
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
        samples = x

    pc = sampler.output_to_point_clouds(samples)[0]
    xyz = pc.coords
    rgb = np.zeros_like(xyz)
    rgb[:,0],rgb[:,1],rgb[:,2] = pc.channels['R'],pc.channels['G'],pc.channels['B']
    return xyz,rgb

def init_from_shapee(prompt):
    from shap_e.diffusion.sample import sample_latents
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config as diffusion_from_config_shape
    from shap_e.models.download import load_model, load_config
    from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
    from shap_e.util.notebooks import decode_latent_mesh

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    model.load_state_dict(torch.load('./load/shapE_finetuned_with_330kdata.pth', map_location=device)['model_state_dict'])
    diffusion = diffusion_from_config_shape(load_config('diffusion'))

    batch_size = 1
    guidance_scale = 15.0
    prompt = str(prompt)
    print('prompt',prompt)

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    render_mode = 'nerf' # you can change this to 'stf'
    size = 512 # this is the size of the renders; higher values take longer to render.

    cameras = create_pan_cameras(size, device)
    shapeimages = decode_latent_images(xm, latents[0], cameras, rendering_mode=render_mode)

    pc = decode_latent_mesh(xm, latents[0]).tri_mesh()

    skip = 4
    coords = pc.verts
    rgb = np.concatenate([pc.vertex_channels['R'][:,None],pc.vertex_channels['G'][:,None],pc.vertex_channels['B'][:,None]],axis=1) 

    coords = coords[::skip]
    rgb = rgb[::skip]
    
    return coords*3.5*0.26,rgb