import requests
from PIL import Image

import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor, LlavaOnevisionForConditionalGeneration, AutoModelForVision2Seq, LlavaForConditionalGeneration

# model_id = "mistral-community/pixtral-12b"
# model = LlavaForConditionalGeneration.from_pretrained(
#     model_id, 
#     torch_dtype=torch.bfloat16, 
#     device_map="auto"
# )

model_id = "Qwen/Qwen2.5-VL-72B-Instruct"
model = AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)

# model_id = "llava-hf/llava-onevision-qwen2-72b-ov-hf"
# model = LlavaOnevisionForConditionalGeneration.from_pretrained(
#     model_id, 
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# model_id = "llava-hf/llava-onevision-qwen2-72b-ov-hf"
# model = LlavaOnevisionForConditionalGeneration.from_pretrained(
#     model_id, 
#     torch_dtype=torch.bfloat16,
#     device_map="auto"
# )

# model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
# model = MllamaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# model = LlavaOnevisionForConditionalGeneration.from_pretrained(
#     model_id, 
#     torch_dtype=torch.float16, 
#     low_cpu_mem_usage=True, 
# )

processor = AutoProcessor.from_pretrained(model_id)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
img_path = '/data/1bali/Other_LLM_projects/multi_view_3DQA/view_e0_a0_numbered_inverted_Mechanical_Parts_Chains_Plate_Wheel_ISO_606_Simplex_½x⅛_Plate_Wheel_simplex_½x⅛.png' #'/data/1bali/Other_LLM_projects/multi_view_3DQA/view_e0_a0_numbered_inverted_Mechanical_Parts_Profiles_EN_EN10219_Rectangular_Hollow_Sections_Rectangular_hollow_section_100x50x8_EN10219_S235JRH.png'#'/data/1bali/Other_LLM_projects/multi_view_3DQA/FreeCAD-library-geometries/Mechanical_Parts_Profiles_EN_EN10219_Rectangular_Hollow_Sections_Rectangular_hollow_section_100x50x8_EN10219_S235JRH/renders_combined/view_e0_a0.png'
az_angle =  int(img_path.split('/')[-1].split('.')[0].split('_a')[1].split('_')[0])
ele_angle = int(img_path.split('/')[-1].split('.')[0].split('_e')[1].split('_')[0])

if ele_angle < 0: 
    el_txt = f"{0-ele_angle} degrees below the x-z plane"
else:
    el_txt = f"{ele_angle} degrees above the x-z plane"
if az_angle < 90:
    az_txt = f"{az_angle} degrees anticlockwise from the front"
elif az_angle==90:
    az_txt = "from the right side"
elif az_angle < 180:
    az_txt = f"{az_angle - 90} degrees anticlockwise from the right side"
elif az_angle == 180:
    az_txt = "from the back side"
elif az_angle < 270:
    az_txt = f"{90 - (az_angle - 180)} degrees clockwise from the left side"
else:
    az_txt = f"{360 - az_angle} degrees clockwise from the front"

## Black background prompt
# prompt = "Here's an image of the object with its Stress Gradient Field in the black background and numbers overlayed on the image." \
# f"The image of the object are currently taken at {az_txt} and {el_txt}."\
# "In the Stress Gradient image, white to red denote high stress gradient regions, while dark blue to blue denote low/no stress gradient regions"\
# "Where are the stress gradients high on the CAD object? Write on what grid cells (their number is written on them) on the image are the stress gradients high(white or yellow or orange or red). Do not give numbers where the stress gradients are low i.e the blue and black areas"

## Black background revised prompt for ranking regions
# prompt = "Here is an image of the object with its Stress Gradient Field in the black background and numbers overlayed on the image." \
# "In the Stress Gradient image, white to yellow to orange to red denote high stress gradient regions (in ascending order), while dark blue to blue denote low/no stress gradient regions"\
# "Where are the stress gradients high on the CAD object?"\
# "Write on what numbers on the image are the stress gradients high and Rank the high stress gradient regions in descending order of intensity starting from Red, Orange, Yellow, White, Light Blue to Dark blue (Highest to lowest)"

## Black background revised prompt for ranking regions based on color with grid
prompt = "Here is an image of the object with its Stress Gradient Field in the black background and numbers and cells overlayed on the image." \
"In the Stress Gradient image, white to yellow to orange to red denote high stress gradient regions (in ascending order), while dark blue to blue denote low/no stress gradient regions"\
"Where are the stress gradients high on the CAD object?"\
"Write on what grid cell numbers on the image are the stress gradients high and Rank the high stress gradient regions in descending order of intensity starting from Red, Orange, Yellow, White, Light Blue to Dark blue (Highest to lowest)"

# Black background revised prompt for ranking regions based on cluster size with grid
prompt = "Here is an image of the object with its Stress Gradient Field in the black background and numbers and cells overlayed on the image." \
"In the Stress Gradient image, white to yellow to orange to red denote high stress gradient regions (in ascending order), while dark blue to blue denote low/no stress gradient regions. "\
"Write on what grid cell numbers on the image are the stress gradients high, cluster the adjacent high stress gradient grid cells together and Rank the high stress gradient clusters in descending order of stress grad cluster size (Largest cluster comprising of many grid cells to smaller cluster comprising of fewer grid cells, all containing cells of high stress gradients with points from white to yellow to red)."

## White background revised prompt for ranking regions based on cluster size with grid
# prompt = "Here is an image of the object with its Stress Gradient Field in the white background and numbers and cells overlayed on the image in green(cell numbers) and red(grid) ." \
# "In the Stress Gradient image, black denotes high stress gradient regions , while white to grey denote low/no stress gradient regions. "\
# "Write on what grid cell numbers on the image are the stress gradients high, cluster the adjacent high stress gradient grid cells together and Rank the high stress gradient clusters in descending order of stress grad cluster size (Largest cluster comprising of many grid cells to smaller cluster comprising of fewer grid cells, all containing cells of high stress gradients with points in black)."

# prompt = "Here is an image of the object with its Stress Gradient Field in the white background and numbers and cells overlayed on the image in green(cell numbers) and red(grid) ." \
# "In the Stress Gradient image, black denotes high stress gradient regions , while white to grey denote low/no stress gradient regions. "\
# "Where are the stress gradients high on the CAD object?"\

# prompt = "Here is an image of the object with its Stress Gradient Field in the white background." \
# "In the Stress Gradient image, black denotes high stress gradient regions , while white to grey denote low/no stress gradient regions. "\
# "Where are the stress gradients high on the CAD object? Give a detailed answer"\

#"Where are the stress gradients high on the CAD object?"\
#"Write the clusters comprising of adjacent grid cells where stress gradients are high and rank these high stress gradient clusters in descending order of stress grad cluster size (Largest cluster comprising of many grid cells to smaller cluster comprising of fewer grid cells, all containing cells of high stress gradients with points from white to yellow to red)."\
#"Write on what numbers on the image are the stress gradients high and Rank the high stress gradient regions in descending order of stress grad cluster size (Largest cluster of white point clouds to smallest cluster of point clouds)"


## White background prompt
# prompt = "Here's an image of the object with its Stress Gradient Field in the white background and numbers overlayed on the image in black." \
# f"The image of the object are currently taken at {az_txt} and {el_txt}."\
# "In the Stress Gradient image, black denotes high stress gradient regions, while white to grey denote low/no stress gradient regions"\
# "Where are the stress gradients high on the CAD object? Write on what grid cell numbers on the image are the stress gradients high(black)."\

conversation = [
    {
      "role": "user",
      "content": [
        {"type": "image"}, 
        {"type": "text", "text": prompt},
        #{"type": "text", "text": "Here's the CAD image of the object and the rest of the views denote the areas in bright colors where there are high stress gradients. Remember that the answer to this question is in the images after the CAD image. Wherever there are high stress gradients (given in brighter colors) there meshing needs to be more refined. Where do we need more refined volumetric mesh? Describe these zones in detail in 3D"}
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

#image_files = ["http://images.cocodataset.org/val2017/000000039769.jpg", 'http://images.cocodataset.org/train2017/000000039770.jpg']
images = [img_path]  # Or use Image.open(requests.get(url, stream=True).raw)

raw_images = [Image.open(image_file) for image_file in images]
inputs = processor(images=raw_images, text=prompt, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=2000, do_sample=False)
print(processor.decode(output[0], skip_special_tokens=True))
