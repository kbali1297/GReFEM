

intro_loading, additional_info = {}, {}
intro_loading['compression'] = (f"A compressive displacement load acts on the object in the -y direction on its top surface as indicated by the blue arrows.\n"
)

intro_loading['bending'] = (f"A bending displacement load acts  on the object in the +y/-y direction on the top surface as indicated by the blue arrows about the horizontal axis indicated via dashed red line.\n"
)

intro_loading['torsion'] = (f"A torsion displacement load acts on the object in the anti-clockwise direction along the x-z plane as indicated by the blue curved arrows about the vertical axis of loading passing through the center of the top surface as indicated by the dashed red line.\n"
)

intro_loading['shear'] = (f"A shear displacement load acts on the object in the +x  direction as indicated by the blue arrows about the vertical axis indicated via dashed red line.\n")

additional_info['compression'] = (f"For the top and bottom views, only consider the BH features. All edges or surfaces along the loading direction are not stress critical as they are not perpendicular to the load and do not create stress concentrations. \n"
f"Neglect the features too far away if the load acts on a narrow part of the top surface passing through the center of the object as they are not stress critical\n"
)

additional_info['bending'] = (f"For the top and bottom views, only consider the BH features. All edges or surfaces along the loading direction are not stress critical as they are not perpendicular to the load and do not create stress concentrations. \n"
f"Geometric features ICE, BH, TH too close to the axis about which load is applied are usually not stress critical, in the case of bending loads features away from the axis are more likely to be stress critical.\n"
)

additional_info['torsion'] = (f"Geometric features ICE, BH, TH too close to the axis about which load is applied are usually not stress critical, in the case of bending loads features away from the axis are more likely to be stress critical.\n")

additional_info['shear'] = (f"Geometric Features ICE, BH, TH that are close to the fixed support are highly likely to be stress critical.\n"
)


loading = ['compression', 'bending', 'torsion', 'shear']

def get_prompt_1(loading_case, prompt_type, num_perspective_views, num_gridded_views):
    if prompt_type == 'geomax':
        return (f"You are given multiple images of a single CAD part.\n\n"
        f"The first {num_perspective_views} images show general 3D views of the object (not gridded).\n"
        f"The remaining {num_gridded_views}+2 images show azimuthal orthographic views and the last two show top and bottom views of the same object.\n"
        f"Each orthographic view has a visible grid with numbered cells.\n\n"
        f"{intro_loading[loading_case]}"
        f"The object is completely fixed at the bottom as indicated by the dashed black lines protruding from the bottom surface\n"
        f"Task:\n"
        f"For EACH gridded orthographic image, identify all grid cells that contain inside them these stress critical features:\n"
        f"1) Internal corner edges (concave edges that fold into the object).\n"
        f"2) Extruded Contour/Portruding Contour (extruded closed contour cut partially like a blind hole (not all the way) into the object or protruding out of the object like a boss)\n"
        f"3) Through Holes (any internal surface with hollow space inside all the way).\n\n"
        f"4) Fillets, or arches (Any concave curved surface)\n"
        f"I.C.E: Internal Corner Edge, E.C/P.C: Extruded Contour/Portruding Contour, T.H: Through Holes, F: Fillets or arches\n"
        f"Use the non-gridded {num_perspective_views} 3D views ONLY to understand the overall shape.\n"
        f"Base all cell predictions ONLY on what is visible in the corresponding gridded orthographic image.\n"
        f"Do NOT guess or infer cells that are not clearly visible in that image.\n\n"
        f"Output format (repeat for each gridded image):\n\n"
        f"***I.C.E: c1, c2, c3 ***\n"
        f"***E.C/P.C: c4, c5 ***\n"
        f"***T.H: c6, c7 ***\n"
        f"***F: c8, c9 ***\n"
        f"{additional_info[loading_case]}"
        f"Rules:\n"
        f"- List only cell numbers.\n"
        f"- If no cells apply, write: NONE.\n"
        f"- DO NOT Add explanations \n")
    elif prompt_type == 'geomid':
        return (f"You are given multiple images of a single CAD part.\n\n"
        f"The first {num_perspective_views} images show general 3D views of the object (not gridded).\n"
        f"The remaining {num_gridded_views}+2 images show azimuthal orthographic views and the last two show top and bottom views of the same object.\n"
        f"Each orthographic view has a visible grid with numbered cells.\n\n"
        f"{intro_loading[loading_case]}"
        f"Task:\n"
        f"For EACH gridded orthographic image, identify all grid cells that contain inside them these stress critical features:\n"
        f"1) Internal corner edges (concave edges that fold into the object).\n"
        f"2) Extruded Contour/Portruding Contour (extruded closed contour cut partially like a blind hole (not all the way) into the object or protruding out of the object like a boss)\n"
        f"3) Through Holes (any internal surface with hollow space inside all the way).\n\n"
        f"4) Fillets, or arches (Any concave curved surface)\n"
        f"I.C.E: Internal Corner Edge, E.C/P.C: Extruded Contour/Portruding Contour, T.H: Through Holes, F: Fillets or arches\n"
        f"Use the non-gridded {num_perspective_views} 3D views ONLY to understand the overall shape.\n"
        f"Base all cell predictions ONLY on what is visible in the corresponding gridded orthographic image.\n"
        f"Do NOT guess or infer cells that are not clearly visible in that image.\n\n"
        f"Output format (repeat for each gridded image):\n\n"
        f"***I.C.E: c1, c2, c3 ***\n"
        f"***E.C/P.C: c4, c5 ***\n"
        f"***T.H: c6, c7 ***\n"
        f"***F: c8, c9 ***\n"
        f"Rules:\n"
        f"- List only cell numbers.\n"
        f"- If no cells apply, write: NONE.\n"
        f"- DO NOT Add explanations \n")
    elif prompt_type == 'geonone':
        return (f"You are given multiple images of a single CAD part.\n\n"
        f"The first {num_perspective_views} images show general 3D views of the object (not gridded).\n"
        f"The remaining {num_gridded_views}+2 images show azimuthal orthographic views and the last two show top and bottom views of the same object.\n"
        f"Each orthographic view has a visible grid with numbered cells.\n\n"
        f"{intro_loading[loading_case]}"
        f"Task:\n"
        f"For EACH gridded orthographic image, identify all grid cells that contain inside them stress critical features.\n"
        f"Use the non-gridded {num_perspective_views} 3D views ONLY to understand the overall shape.\n"
        f"Base all cell predictions ONLY on what is visible in the corresponding gridded orthographic image.\n"
        f"Do NOT guess or infer cells that are not clearly visible in that image.\n\n"
        f"Output format (repeat for each gridded image):\n\n"
        f"***Cells: c1, c2, c3 ***\n"
        f"Rules:\n"
        f"- List only cell numbers.\n"
        f"- If no cells apply, write: NONE.\n"
        f"- DO NOT Add explanations \n")
        
        
def get_prompt_2(num_perspective_views, num_gridded_views):
    return (f"In these views azimuthal orthographic {num_gridded_views} views, filter out cells contain points that are less likely to be stress critical.\n"
            f"Use the non-gridded {num_perspective_views} 3D views ONLY to understand the overall shape.\n"
            f"Loading Case based Info:\n"
            f"1) Compression: On concave surfaces with stress critical anchor points marked, stress critical points are likely to occur across the mid central band.\n"
            f"2) Bending: On concave surfaces with stress critical anchor points marked, stress critical points are likely to occur away from the axis of bending and closer to the outer surface.\n"
            f"3) Torsion: On concave surfaces with stress critical anchor points marked, stress critical points are likely to occur away from the axis of torsion and closer to the outer surface.\n"
            f"4) Shear: On concave surfaces with stress critical anchor points marked, stress critical points are likely to occur closer to the fixed support.\n"
            f"Task:\n"
            f"Base all cell predictions ONLY on what is visible in the corresponding gridded orthographic image.\n"
            f"Do NOT guess or infer cells that are not clearly visible in that image.\n\n"
            f"Output format (repeat for each gridded image):\n\n"
            f"***Cells: c1, c2, c3 ***\n"
            f"Rules:\n"
            f"- List only cell numbers.\n"
            f"- If no cells apply, write: NONE.\n"
            f"- DO NOT Add explanations \n")