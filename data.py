import shutil
from cv2 import line
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



import os
import pickle # For caching
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms

class OrthoViewDataset(Dataset):
    def __init__(self, dataset_list_path, transform=None, cache_path=None):
        self.samples = []
        
        # 1. Try to load from cache first
        if cache_path and os.path.exists(cache_path):
            print(f"Loading dataset metadata from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.samples = pickle.load(f)
        else:
            # Pre-calculate constants
            n_azimuth = 12
            elevations = [-90, -72, -54, -36, -18, 0, 18, 36, 54, 72, 90]
            azimuths = [(360 / n_azimuth) * a for a in range(n_azimuth)]
            
            with open(dataset_list_path, 'r') as fread:
                cad_folders = [line.strip() for line in fread if line.strip()]

            for cad_folder in tqdm(cad_folders, desc="Processing CAD folders"):
                image_paths = []
                labels_dict, labels = {}, []
                
                # Path to the render folder
                render_dir = os.path.join(cad_folder, "renders_pyvista_mesh_initial")
                ortho_views_path = os.path.join(cad_folder, 'ortho_views.txt')

                # 2. Parse labels from txt
                if os.path.exists(ortho_views_path):
                    with open(ortho_views_path, 'r') as f:
                        for line in f:
                            view_str, num_points = line.strip().split(':')
                            #label = 1 if int(num_points) > 0 else 0
                            label = int(num_points)

                            # Logic to extract angles (kept as per your original code)
                            el_angle = int(view_str.split('_e')[1].split('.')[0].split('_')[0])
                            az_angle = int(view_str.split('_a')[1].split('.')[0].split('_')[0])

                            img_name = f"view_e{el_angle}_a{az_angle}.png"
                            labels_dict[img_name] = label
                            

                # 3. Add extra views without calling os.listdir
                for el in elevations:
                    for az in azimuths:
                        img_name = f"view_e{int(el)}_a{int(az)}.png"
                        image_paths.append(os.path.join(render_dir, img_name))
                        labels.append(labels_dict.get(img_name, 0))  # Default to 0 if not found
                        if el in [-90, 90]: # Top/Bottom logic
                            break
                
                self.samples.append({
                    'cad_id': os.path.basename(cad_folder),
                    'image_paths': image_paths,
                    'labels': labels
                })

            # Save cache for next time
            if cache_path:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.samples, f)

        # Transforms (standardized)
        self.transform = transform or transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        imgs, labels, file_names = [], [], []

        for i, img_path in enumerate(s['image_paths']):
            # Efficiency tip: Use try-except here in case a file is missing
            try:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
                labels.append(s['labels'][i])
                file_names.append(os.path.basename(img_path))
            except Exception as e:
                continue 

        return {
            "images": torch.stack(imgs, dim=0),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "cad_id": s["cad_id"],
            "file_order": file_names
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

        

