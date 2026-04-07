import cv2
import numpy as np

# ==== CONFIGURATION ====
#/data/1bali/Other_LLM_projects/multi_view_3DQA/FreeCAD-library-geometries/Mechanical_Parts_Profiles_EN_EN10058_Flat_steel_bars_Flat_Bar_30x8_EN10058_S235JR/renders_pyvista/view_e20_a150_no_colorbar.png
image_path = "/data/1bali/Other_LLM_projects/multi_view_3DQA/FreeCAD-library-geometries/Mechanical_Parts_Chains_Plate_Wheel_ISO_606_Simplex_½x⅛_Plate_Wheel_simplex_½x⅛/renders_pyvista/view_e0_a0_no_colorbar.png"      # your stress image
fname = image_path.split('/')[-3]
output_path = f"view_e0_a0_numbered_inverted_{fname}.png"
n_rows, n_cols = 10, 10              # grid resolution
font_scale = 1.0                   # label size
font_thickness = 2
font_color = (255,255,255) # white #(0, 255, 0)       # green
grid_color = (255,255,255) # white #(0, 0, 255)       # red grid
line_thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX

# ==== LOAD IMAGE ====
img = cv2.imread(image_path)
h, w, _ = img.shape

# ==== DRAW GRID ====
cell_h = h // n_rows
cell_w = w // n_cols

number = 1
for r in range(n_rows):
    for c in range(n_cols):
        y0, y1 = r * cell_h, (r + 1) * cell_h
        x0, x1 = c * cell_w, (c + 1) * cell_w

        # Draw cell rectangle
        cv2.rectangle(img, (x0, y0), (x1, y1), grid_color, line_thickness)

        # Center coordinates for text
        text = str(number)
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = x0 + (cell_w - text_size[0]) // 2
        text_y = y0 + (cell_h + text_size[1]) // 2

        # Draw text with outline for visibility
        cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness + 3, cv2.LINE_AA)
        cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        number += 1

# ==== SAVE OUTPUT ====
cv2.imwrite(output_path, img)
print(f"Saved numbered image as {output_path}")
