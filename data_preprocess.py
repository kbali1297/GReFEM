import os
import numpy as np

if __name__ == "__main__":
    log_file_path = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/generate_orthoviews.log'

    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    valid_cads, invalid_cads = [], []
    for line in lines:
        if '[DONE]' in line:
            valid_cad_id = line.split('[DONE] ')[1].strip()
            valid_cads.append(valid_cad_id)
        elif '[SKIP]' in line or '[ERROR]' in line:
            invalid_cad_id = line.split('] ')[1].strip()
            invalid_cads.append(invalid_cad_id)
    
    print(f'Valid CADs: {len(valid_cads)}')
    print(f'Invalid CADs: {len(invalid_cads)}')
    print('Lets See')

    # with open('dataset.log', 'w') as f:
    #     for cad in valid_cads:
    #         f.write(f'/data/1bali/Other_LLM_projects/ECCV_2026/ABC_CAD_Dataset_small2/{cad}\n')

    valid_cads = sorted(valid_cads)

    seed = 42
    test_set_dir = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/test_meshes_7.04.2026'
    test_cads = set(os.listdir(test_set_dir))
    train_cads = [cad for cad in valid_cads if cad not in test_cads]
    
    np.random.seed(seed)
    
    with open('/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/generate_renders_26.04.2026.log', 'r') as f:
        lines = f.readlines()
    for line in lines:
        if 'is empty or invalid' in line:
            cad_id = line.strip().split('/')[-1].split('_')[0]
            if cad_id in train_cads:
                train_cads.remove(cad_id)
            if cad_id in test_cads:
                test_cads.remove(cad_id)
    
    with open('train.log', 'w') as f:
        for cad in train_cads:
            f.write(f'/data/1bali/Other_LLM_projects/ECCV_2026/ABC_CAD_Dataset_small2/{cad}\n')

    with open('val.log', 'w') as f:
        for cad in test_cads:
            f.write(f'/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/test_meshes_7.04.2026/{cad}\n')