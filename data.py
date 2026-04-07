import shutil
from tqdm import tqdm
import os, sys
import numpy as np
import trimesh
#freecad_base = os.environ["CONDA_PREFIX"]

# Append FreeCAD's Python library paths
# sys.path.append(os.path.join(freecad_base, "lib"))   # core FreeCAD libraries
# sys.path.append(os.path.join(freecad_base, "Mod"))   # FreeCAD Python modules (Part, Mesh, etc.)

# # Now you can import FreeCAD normally
# import FreeCAD
# import Part
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



class OrthoViewDataset(Dataset):
    def __init__(self, dataset_list_path, transform=None):
        """
        image_dataset: dict[cad_id -> list of view dicts]
        """
        cad_folders = []
        with open(dataset_list_path, 'r') as fread:
            for line in fread:
                line = line.strip()
                cad_folders.append(line)

        self.samples = []
        for cad_folder in tqdm(cad_folders,
                        total=len(cad_folders),
                        desc="Processing CAD folders",
                        unit="folder"):

            ## Generate labels
            ortho_views_path = f'{cad_folder}/ortho_views.txt'

            with open(ortho_views_path, 'r') as f:
                image_paths, labels = [], []
                for line in f:
                    line = line.strip()
                    try:
                        view_str, num_views = line.split(':')
                        if int(num_views)>0: label=1
                        else: label=0
                        el_angle = view_str.split('_e')[1].split('.')[0]
                        az_angle = view_str.split('_a')[1].split('.')[0]
                        image_paths.append(f"{cad_folder}/"
                                                "renders_pyvista_mesh_initial/"
                                                f"view_e{int(el_angle)}_a{int(az_angle)}.png")
                        labels.append(label)
                    except: pass

            ## Adding extra views to choose better
            for img_name in os.listdir(f'{cad_folder}/renders_pyvista_mesh_initial'):
                n_elevation, n_azimuth = 9, 12
                if img_name.startswith('view_e') and img_name not in image_paths:
                    el_angle = img_name.split('_e')[1].split('.')[0]
                    az_angle = img_name.split('_a')[1].split('.')[0]
                    elevations = [-90 + (180 /(n_elevation+1)) * e for e in range(1, n_elevation+1)]
                    azimuths = [(360 / n_azimuth) * a for a in range(n_azimuth)]
                    if el_angle in elevations and az_angle in azimuths:
                        image_paths.append(f"{cad_folder}/"
                                            f"renders_pyvista_mesh_initial/{img_name}")
                        labels.append(0)
            
            self.samples.append(
                {'cad_id': os.path.basename(cad_folder),
                'image_paths': image_paths,
                'labels': labels})

        ## Dino specific transform composition
        # self.try_transform = transforms.Compose([
        #     transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        #     transforms.CenterCrop(224)])
        
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx=None):
        
        s = self.samples[idx]


        ## Randomize image order so the network is agnostic to the view ordering and learns to focus on views alone
        perm = torch.randperm(len(s['labels']))

        imgs, labels, perm_image_paths = [],[], []
        for perm_idx in perm:
            img_path = s['image_paths'][perm_idx]
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)

            label = torch.tensor(s['labels'][perm_idx], dtype=torch.float32)
            imgs.append(img)
            labels.append(label)
            perm_image_paths.append(os.path.basename(img_path))


        # Stack into tensors
        images_tensor = torch.stack(imgs, dim=0)           # (V, 3, H, W)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)  # (V,)

        return {
            "images": images_tensor,
            "labels": labels_tensor,
            "cad_id": s["cad_id"],
            "file_order": perm_image_paths
        }


if __name__ == '__main__':
    
    ## Prepare train/val split
    dataset_path = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/dataset'
    cad_folders = os.listdir(dataset_path)
    
    cad_arr = np.array(cad_folders)
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(len(cad_arr))

    shuffled_cad_arr = cad_arr[perm]

    split_idx = int(0.95 * len(shuffled_cad_arr))
    split = {}
    split['train'] = shuffled_cad_arr[:split_idx]
    split['val'] = shuffled_cad_arr[split_idx:]
    #split['overfit'] = shuffled_cad_arr[-5:]

    ## Write them in text files
    for split_key in ['train', 'val']:#, 'overfit']: 
        with open(f'{os.path.dirname(dataset_path)}/{split_key}.txt', 'w') as f_split:
            for cad_folder in split[split_key]:
                f_split.write(f'{os.path.dirname(dataset_path)}/dataset/{cad_folder}\n')

    train_dataset = OrthoViewDataset(f'{os.path.dirname(dataset_path)}/train.txt')
    val_dataset = OrthoViewDataset(f'{os.path.dirname(dataset_path)}/val.txt')
    #overfit_dataset = OrthoViewDataset(f'{os.path.dirname(dataset_path)}/overfit.txt')

    print(train_dataset[0])
    print(val_dataset[5])
    #print(overfit_dataset[-1])
    print('Lets check!')

        

