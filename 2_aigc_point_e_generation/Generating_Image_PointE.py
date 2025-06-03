import torch
import os
import numpy as np
import random
from tqdm.auto import tqdm
from collections import defaultdict
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

# Prompt templates for each class to generate diverse textual descriptions
prompt_templates = [
    "a 3D point cloud of a {}",
    "a realistic 3D scan of a {}",
    "a synthetic point cloud of a {} on a table",
    "a low-poly 3D model of a {}",
    "a 3D point cloud of a {} from above",
    "a point cloud of a {} rendered in a virtual scene",
    "an artistic 3D rendering of a {}",
    "a point cloud of a {} from a side view",
    "a textured 3D scan of a {} in a room",
    "a simplified point cloud version of a {}"
]


# Load point cloud from .off file
def load_off(file_path):
    with open(file_path, 'r') as f:
        if 'OFF' != f.readline().strip():
            raise ValueError('Not a valid .off file')
        n_verts, n_faces, _ = map(int, f.readline().strip().split())
        verts = []
        for _ in range(n_verts):
            verts.append(list(map(float, f.readline().strip().split()[:3])))

        return np.array(verts)


# Load the ModelNet40 dataset (.off format) into memory
def load_modelnet40(dataset_path):
    dataset = {}
    file_names_dict = {}

    for class_name in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        dataset[class_name] = []
        file_names_dict[class_name] = []

        for file_name in sorted(os.listdir(class_path)):
            if file_name.endswith('.off'):
                file_path = os.path.join(class_path, file_name)
                point_cloud = load_off(file_path)

                if point_cloud is not None:
                    dataset[class_name].append(point_cloud)
                    file_names_dict[class_name].append(file_name[:-4])  # remove .off extension

    return dataset, file_names_dict


# Retrieve class names (folder names) from dataset directory
def get_modelnet40_class_names(dataset_path):
    class_names = []
    for name in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, name)):
            class_names.append(name)
    return sorted(class_names)


# Generate 10 prompt variants for a given class name
def generate_class_prompts(class_name):
    display_name = class_name
    return [template.format(display_name) for template in prompt_templates]


# Generate point clouds for all classes and save results
def generate_for_all_classes(sampler, class_names, all_prompts_dict, output_dir):
    for class_name in class_names:
        prompts = all_prompts_dict.get(class_name, [])
        class_dir = os.path.join(str(output_dir), str(class_name))
        os.makedirs(class_dir, exist_ok=True)
        generate_class_samples(sampler, prompts, class_dir, class_name)


# Generate 25 samples per class using grouped prompt strategy
def generate_class_samples(sampler, prompts, class_dir, class_name):
    random.shuffle(prompts)
    group1, group2 = prompts[:5], prompts[5:]
    sample_index = 0
    file_records = []

    # Group 1: 5 prompts × 2 samples = 10
    for prompt in group1:
        for _ in range(2):
            output_path = os.path.join(class_dir, f"{class_name}_{sample_index:04d}.ply")
            pc = generate_pc(sampler, prompt)
            with open(output_path, "wb") as f:
                pc.write_ply(f)
            file_records.append(f"[G1] {os.path.basename(output_path)} | {prompt[:30]}...")
            sample_index += 1

    # Group 2: 5 prompts × 3 samples = 15
    for prompt in group2:
        for _ in range(3):
            output_path = os.path.join(class_dir, f"{class_name}_{sample_index:04d}.ply")
            pc = generate_pc(sampler, prompt)
            with open(output_path, "wb") as f:
                pc.write_ply(f)
            file_records.append(f"[G2] {os.path.basename(output_path)} | {prompt[:30]}...")
            sample_index += 1

    print(f"\n=== Class {class_name} generating ===")
    for record in file_records:
        print(record)
    print(f"✅ Done. Total samples: {sample_index}/25 | Saved to: {class_dir}\n")


# Generate a single point cloud from text prompt using the diffusion model
def generate_pc(sampler, prompt):
    samples = None
    for x in sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt])):
        samples = x
    return sampler.output_to_point_clouds(samples)[0]


# Entry point for model setup and sample generation
if __name__ == "__main__":
    device = torch.device('cuda')

    # Load base model
    print('Creating base model...')
    base_name = 'base40M-textvec'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    # Load upsampler model
    print('Creating upsampler model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    # Download and load checkpoints
    print('Downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print('Downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    # Initialize sampler
    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],  # Two-stage sampling: base + upsample
        aux_channels=['R', 'G', 'B'],  # Add RGB to output
        guidance_scale=[3.0, 0.0],  # Text guidance applied only at base stage
        model_kwargs_key_filter=('texts', ''),
    )

    # Dataset and output directory paths
    dataset_path = "ModelNet40_sampled/ModelNet40_sampled"
    output_path = "ModelNet40_generated_v2"

    # Load class names and generate prompts
    dataset, file_names_dict = load_modelnet40(dataset_path)
    class_names = get_modelnet40_class_names(dataset_path)
    prompts = {class_name: generate_class_prompts(class_name) for class_name in class_names}

    # Generate 25 point cloud samples for each class
    generate_for_all_classes(sampler, class_names, prompts, output_path)
