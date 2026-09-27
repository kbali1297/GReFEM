from data import *
from model import *
from generate_renders import *
from pathlib import Path
import torch.nn.functional as F
import math
import re
import os

@torch.no_grad()
def non_redundant_views(
    sorted_image_paths,
    transform,
    device,
    sim_thresh=0.9,
):
    """
    Greedy non-redundant view selection using image similarity (DINO).
    Assumes sorted_image_paths are ordered best → worst.
    Returns full image paths.
    """
    dino_encoder = DinoEncoder().to(device)
    dino_encoder.eval()

    imgs =[]
    for img_path in sorted_image_paths:
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        imgs.append(img)

    images_tensor = torch.stack(imgs).to(device)
    embeddings = dino_encoder(images_tensor)
    embeddings = F.normalize(embeddings, dim=-1)

    kept_indices =[]

    for i in range(len(sorted_image_paths)):
        if not kept_indices:
            kept_indices.append(i)
            continue

        sims = embeddings[i] @ embeddings[kept_indices].T
        if torch.all(sims < sim_thresh):
            kept_indices.append(i)

    return [sorted_image_paths[i] for i in kept_indices]


def extract_az_el(file_path):
    """
    Helper function to extract azimuth and elevation from the file path.
    Modify the regex below if your generated filename format differs.
    """
    filename = os.path.basename(file_path)
    
    # Try finding explicit az and el keywords (e.g. render_az_120_el_30.png)
    match = re.search(r'(?:az|azimuth)[^\d\-]*([-+]?\d*\.?\d+).*?(?:el|elevation)[^\d\-]*([-+]?\d*\.?\d+)', filename, re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))
        
    # Fallback: extract the first two numerical patterns found in the filename
    numbers = re.findall(r'[-+]?\d*\.?\d+', filename)
    if len(numbers) >= 2:
        return float(numbers[0]), float(numbers[1])
        
    raise ValueError(f"Could not extract azimuth and elevation from {filename}. Please modify the extract_az_el() regex.")


def angle_between(az1, el1, az2, el2):
    """
    Calculate the actual spherical angle between two views in degrees.
    Assumes azimuth and elevation are given in degrees, and elevation is 
    the angle from the horizontal plane (0 at equator, 90 at pole).
    """
    az1_r, el1_r = math.radians(az1), math.radians(el1)
    az2_r, el2_r = math.radians(az2), math.radians(el2)
    
    # Spherical to Cartesian representation
    x1, y1, z1 = math.cos(el1_r) * math.cos(az1_r), math.cos(el1_r) * math.sin(az1_r), math.sin(el1_r)
    x2, y2, z2 = math.cos(el2_r) * math.cos(az2_r), math.cos(el2_r) * math.sin(az2_r), math.sin(el2_r)
    
    # Dot product gives the cosine of the angle between them
    dot_product = x1*x2 + y1*y2 + z1*z2
    
    # Clamp to [-1.0, 1.0] to avoid float precision issues with math.acos
    dot_product = max(min(dot_product, 1.0), -1.0)
    
    angle_rad = math.acos(dot_product)
    return math.degrees(angle_rad)


def angle_based_nms(sorted_image_paths, min_angle_diff=30.0):
    """
    Greedy Non-Maximum Suppression based on spherical angles.
    Views closer than min_angle_diff to any already kept view are suppressed.
    """
    kept_paths =[]
    kept_angles =[] # Will store tuples of (az, el)
    
    for img_path in sorted_image_paths:
        az, el = extract_az_el(img_path)
        
        # Check angle distance against all currently kept views
        is_redundant = False
        for (k_az, k_el) in kept_angles:
            angle = angle_between(az, el, k_az, k_el)
            if angle < min_angle_diff:
                is_redundant = True
                break
                
        if not is_redundant:
            kept_paths.append(img_path)
            kept_angles.append((az, el))
            
    return kept_paths


def infer_NN(mesh_file_path, model_chkpt_path, choose_top=3, output_dirpath=None, min_angle_diff=30.0):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dir = '/'.join(mesh_file_path.split('/')[:-3])
    cad_id = os.path.basename(mesh_file_path).replace(".obj", "")
    if output_dirpath is None:
        output_dirpath = f"{test_dir}/{cad_id}/renders_pyvista_mesh_initial"

    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    imgs, image_paths = [],[]
    
    n_azimuth = 12
    elevations = [-90, -72, -54, -36, -18, 0, 18, 36, 54, 72, 90]
    azimuths = [(360 / n_azimuth) * a for a in range(n_azimuth)]
            
    for el in elevations:
        for az in azimuths:
            img_name = f"view_e{int(el)}_a{int(az)}.png"
            img_path = os.path.join(output_dirpath, img_name)
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                img = transform(img)
                imgs.append(img)
                image_paths.append(img_path)   # ✅ FULL PATH
            else:
                print(f"Warning: Expected image {img_path} not found. Please check your render generation step.")
            
            if el in [-90, 90]: # Top/Bottom logic
                break
    # for img_name in os.listdir(output_dirpath):
    #     img_path = os.path.join(output_dirpath, img_name)
    #     img = Image.open(img_path).convert("RGB")
    #     img = transform(img)
    #     imgs.append(img)
    #     image_paths.append(img_path)   # ✅ FULL PATH

    images_tensor = torch.stack(imgs).to(device)

    # ------------------------
    # Model inference
    # ------------------------
    ortho_selector_model = DinoViewSelector(num_views=110).to(device)
    ortho_selector_model.load_state_dict(torch.load(model_chkpt_path))
    ortho_selector_model.eval()

    with torch.no_grad():
        scores = ortho_selector_model(images_tensor.unsqueeze(0))[0]

    # ------------------------
    # Sort by score
    # ------------------------
    idxs_sorted = torch.argsort(scores, descending=True).cpu().tolist()
    sorted_image_paths = [image_paths[i] for i in idxs_sorted]

    # ------------------------
    # Remove redundancy (Angle-based NMS)
    # ------------------------
    non_redundant = angle_based_nms(
        sorted_image_paths,
        min_angle_diff=min_angle_diff
    )

    # ------------------------
    # Return top-K FULL PATHS
    # ------------------------
    return non_redundant[:choose_top]
    

if __name__ == '__main__':

    test_dir = './test_meshes'
    flag=0
    for cad_folder in os.listdir(test_dir):
        # if flag==0 and cad_folder!='00200022': 
        #     continue
        # else: flag=1
        print(f"\nProcessing CAD: {cad_folder}")
        mesh_file_path = os.path.join(test_dir, cad_folder, 'renders_pyvista', f'{cad_folder}.obj')
        model_chkpt_path = './model_saves/ep5_val0.0338.pth'

        chosen_views = infer_NN(mesh_file_path, model_chkpt_path, choose_top=10, min_angle_diff=30.0)
        
        print(f'CAD: {cad_folder} | Selected Top Views:\n')
        # for view in chosen_views:
        #     print(view)

        with open(f'{test_dir}/{cad_folder}/pred_ortho_views2.log', 'w') as f:
            for view in chosen_views:
                f.write(f"{view}\n")