import torch
import os
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config

# 🔹 Set test prompt and output path
prompt = "a 3D point cloud of a chair"
output_dir = "./test_generated"
os.makedirs(output_dir, exist_ok=True)

# 🔹 Initialize device
device = torch.device('cuda')

# 🔹 Load base and upsampler models
print('Loading models...')
base_name = 'base40M-textvec'
base_model = model_from_config(MODEL_CONFIGS[base_name], device).eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device).eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

# 🔹 Load model checkpoints
base_model.load_state_dict(load_checkpoint(base_name, device))
upsampler_model.load_state_dict(load_checkpoint('upsample', device))

# 🔹 Initialize sampler
sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0, 0.0],
    model_kwargs_key_filter=('texts', ''),
)

# 🔹 Generate a single point cloud sample
print('Generating point cloud...')
samples = None
for x in sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt])):
    samples = x

point_cloud = sampler.output_to_point_clouds(samples)[0]

# 🔹 Save the generated point cloud
output_path = os.path.join(output_dir, "test_sample.ply")
with open(output_path, "wb") as f:
    point_cloud.write_ply(f)

print(f"✅ Sample saved to: {output_path}")
