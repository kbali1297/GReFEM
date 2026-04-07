from data import *
from model import *
from generate_renders import *
from pathlib import Path
import torch.nn.functional as F

@torch.no_grad()
def non_redundant_views(
    sorted_image_paths,
    transform,
    device,
    sim_thresh=0.9,
):
    """
    Greedy non-redundant view selection.
    Assumes sorted_image_paths are ordered best → worst.
    Returns full image paths.
    """

    dino_encoder = DinoEncoder().to(device)
    dino_encoder.eval()

    imgs = []
    for img_path in sorted_image_paths:
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        imgs.append(img)

    images_tensor = torch.stack(imgs).to(device)
    embeddings = dino_encoder(images_tensor)
    embeddings = F.normalize(embeddings, dim=-1)

    kept_indices = []

    for i in range(len(sorted_image_paths)):
        if not kept_indices:
            kept_indices.append(i)
            continue

        sims = embeddings[i] @ embeddings[kept_indices].T
        if torch.all(sims < sim_thresh):
            kept_indices.append(i)

    return [sorted_image_paths[i] for i in kept_indices]


def infer_NN(mesh_file_path, model_chkpt_path, choose_top=3, output_dirpath=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path_curdir = Path(__file__).resolve().parent
    cad_id = os.path.basename(mesh_file_path).replace(".obj", "")

    if output_dirpath is None:
        output_dirpath = f"{path_curdir}/{cad_id}/renders_pyvista_mesh_initial"

    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    imgs, image_paths = [], []
    for img_name in os.listdir(output_dirpath):
        img_path = os.path.join(output_dirpath, img_name)
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        imgs.append(img)
        image_paths.append(img_path)   # ✅ FULL PATH

    images_tensor = torch.stack(imgs).to(device)

    # ------------------------
    # Model inference
    # ------------------------
    ortho_selector_model = DinoViewSelector().to(device)
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
    # Remove redundancy
    # ------------------------
    # non_redundant = non_redundant_views(
    #     sorted_image_paths,
    #     transform,
    #     device,
    #     sim_thresh=0.98,
    # )

    # ------------------------
    # Return top-K FULL PATHS
    # ------------------------
    return sorted_image_paths[:choose_top] #non_redundant[:choose_top]
    

if __name__ == '__main__':

    #'/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/test_meshes/Mechanical_Parts_Carpentry_iron_profile_Perfil_R_5853/renders_pyvista/Mechanical_Parts_Carpentry_iron_profile_Perfil_R_5853.obj'
    #'/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/test_meshes/Industrial_Design_Shelf_Cloud_shelf/renders_pyvista/Industrial_Design_Shelf_Cloud_shelf.obj'#
    #'/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/test_meshes/Electronics_Parts_Connectors_dupont-connectors_dupont-2_54mm-female-conn-1x1/renders_pyvista/Electronics_Parts_Connectors_dupont-connectors_dupont-2_54mm-female-conn-1x1.obj'
    #mesh_file_path = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/test_meshes/Electrical_Parts_Servos_SG-90_SG90-1-arm-horn/renders_pyvista/Electrical_Parts_Servos_SG-90_SG90-1-arm-horn.obj'#'/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/test_meshes/Generic_objects_Scale_Models_Cement_mixer_truck_cabin_back/renders_pyvista/Generic_objects_Scale_Models_Cement_mixer_truck_cabin_back.obj
    mesh_file_path = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/dataset/inverted_l/renders_pyvista/inverted_l.obj'
    model_chkpt_path = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/model_saves/ortho_view_selector_20.pth'

    chosen_views = infer_NN(mesh_file_path, model_chkpt_path)
    print(f'Lets See')
    



    

