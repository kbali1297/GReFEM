import os

def extract_el_az_from_viewpath(view_path):
    # Example view_path: '.../renders/view_e-30_a45.png'
    filename = os.path.basename(view_path)
    name_part = filename.split('.')[0]  # 'view_e-30_a45'
    
    # Extract el and az using string manipulation
    el_str = name_part.split('_e')[1].split('_')[0]  # '-30'
    az_str = name_part.split('_a')[1].split('_')[0]  # '45'
    
    return int(el_str), int(az_str)

intro_loading, additional_info, info_load_filtering = {}, {}, {}
intro_loading['compression'] = (f"A compressive displacement load acts on the object in the -y direction on its top surface as indicated by the blue arrows.\n"
)

intro_loading['bending'] = (f"A bending displacement load acts  on the object in the +y/-y direction on the top surface as indicated by the blue arrows about the horizontal axis indicated via dashed red line.\n"
)

intro_loading['torsion'] = (f"A torsion displacement load acts on the object in the anti-clockwise direction along the x-z plane as indicated by the blue curved arrows about the vertical axis of loading passing through the center of the top surface as indicated by the dashed red line.\n"
)

intro_loading['bending_compression'] = (f"A bending displacement load acts  on the object in the +y/-y direction on the top surface as indicated by the blue arrows about the horizontal axis indicated via dashed red line. Also a "
f"compressive load acts on the object in the +/- x direction as shown by the blue arrows\n"
)

intro_loading['torsion_compression'] = (f"A torsion displacement load acts on the object in the anti-clockwise direction along the x-z plane as indicated by the blue curved arrows about the vertical axis of loading passing through the center of the top surface as indicated by the dashed red line. Also the compressive load acts on the object in the +/- x direction as shown by the blue arrows\n")

#intro_loading['shear'] = (f"A shear displacement load acts on the object in the +x  direction as indicated by the blue arrows about the vertical axis indicated via dashed red line.\n")

additional_info['compression'] = (f"For the top and bottom views, only consider the BH features. All edges or surfaces along the loading direction are not stress critical as they are not perpendicular to the load and do not create stress concentrations. \n"
f"Neglect the features too far away if the load acts on a narrow part of the top surface passing through the center of the object as they are not stress critical\n"
)

additional_info['bending'] = (f"For the top and bottom views, only consider the BH features. All edges or surfaces along the loading direction are not stress critical as they are not perpendicular to the load and do not create stress concentrations. \n"
f"Geometric features ICE, BH, TH too close to the axis about which load is applied are usually not stress critical, in the case of bending loads features away from the axis are more likely to be stress critical.\n"
)

additional_info['torsion'] = (f"Geometric features ICE, BH, TH too close to the axis about which load is applied are usually not stress critical, in the case of torsion loads features away from the axis are more likely to be stress critical.\n")

additional_info['bending_compression'] = (f"For each load (bending or compression), only consider the BH features when viewed parallel to the loading direction. All edges or surfaces along the loading direction for each load are not stress critical but perpendicular to the respective loads are. \n"   
f"Geometric features ICE, BH, TH too close to the axis about which bending load is applied are usually not stress critical, in the case of bending loads features away from the axis are more likely to be stress critical. Consider stress critical areas taking care of both the loads\n"
)   

additional_info['torsion_compression'] = (f"For the torsion load geometric features ICE, BH, TH too close to the axis about which load is applied are usually not stress critical, in the case of torsion loads features away from the axis are more likely to be stress critical. Also, features along the loading direction of the compressive load are not stress critical as they do not create stress concentrations."
                                          f"Consider stress critical areas taking care of both the loads.\n"
)
# additional_info['shear'] = (f"Geometric Features ICE, BH, TH that are close to the fixed support are highly likely to be stress critical.\n"
# )


loading = ['compression', 'bending', 'torsion', 'shear']

def get_prompt_1(loading_case, prompt_type, perspective_views, gridded_views):
    
    num_gridded_views = len(gridded_views)
    num_perspective_views = len(perspective_views)

    angles_list_gridded_views, angles_list_perspective_views = [], []
    for view_path in gridded_views:
        el, az = extract_el_az_from_viewpath(view_path)
        angles_list_gridded_views.append((f'e{el}', f'a{az}'))
    for view_path in perspective_views:
        el, az = extract_el_az_from_viewpath(view_path)
        angles_list_perspective_views.append((f'e{el}', f'a{az}'))
    if prompt_type == 'geo_max':
        return (f"You are given multiple images of a single CAD part along with their camera angles marked in (e: elevation, a: azimuth) format. These angles symbolize the location of the camera with respect to the objects' center in spherical coordinates.\n\n"
        f"The first {num_perspective_views} images show general 3D views of the object (not gridded) with camera situated at various angles respectively: {angles_list_perspective_views}.\n"
        f"The remaining {num_gridded_views} images show orthographic views with camera positions in the list respectively (e: elevation, a: azimuth): {angles_list_gridded_views}.\n"
        f"Each orthographic view has a visible grid with numbered cells.\n\n"
        f"{intro_loading[loading_case]}"
        f"The object is completely fixed at the bottom as indicated by the dashed black lines protruding from the bottom surface\n"
        f"Task:\n"
        f"For EACH gridded orthographic image, identify all grid cells that contain inside them these features that will be stress critical for the given loading condition:\n"
        f"1) Internal corner edges (concave edges that fold into the object and go into the view plane).\n"
        f"2) Extruded Contour/Portruding Contour (extruded closed contour cut partially like a blind hole (not all the way) into the object or protruding out of the object like a boss)\n"
        f"3) Through Holes (any internal surface with hollow space inside all the way).\n\n"
        f"4) Concave Fillet Surfaces, or arches (Any concave curved surface)\n"
        f"I.C.E: Internal Corner Edge, E.C/P.C: Extruded Contour/Portruding Contour, T.H: Through Holes, F: Concave Fillet Surfaces or arches\n"
        f"Use the non-gridded {num_perspective_views} 3D views ONLY to understand the overall shape. The grid numbers should be marked only for the corresponding gridded orthographic images.\n"
        f"Base all cell predictions ONLY on what is visible in the corresponding gridded orthographic image.\n"
        f"For Hollow features mark the boundary cells that contain the hollow feature inside them.\n"
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
    elif prompt_type == 'geo_mid':
        return (f"You are given multiple images of a single CAD part along with their camera angles marked in (e: elevation, a: azimuth) format. These angles symbolize the location of the camera with respect to the objects' center in spherical coordinates.\n\n"
        f"The first {num_perspective_views} images show general 3D views of the object (not gridded) with camera situated at various angles respectively: {angles_list_perspective_views}.\n"
        f"The remaining {num_gridded_views} images show orthographic views with camera positions in the list respectively (e: elevation, a: azimuth): {angles_list_gridded_views}.\n"
        f"Each orthographic view has a visible grid with numbered cells.\n\n"
        f"{intro_loading[loading_case]}"
        f"Task:\n"
        f"For EACH gridded orthographic image, identify all grid cells that contain inside them these features that will be stress critical for the given loading condition:\n"
        f"1) Internal corner edges (concave edges that fold into the object and go into the plane).\n"
        f"2) Extruded Contour/Portruding Contour (extruded closed contour cut partially like a blind hole (not all the way) into the object or protruding out of the object like a boss)\n"
        f"3) Through Holes (any internal surface with hollow space inside all the way).\n\n"
        f"4) Concave Fillet Surfaces, or arches (concave curved surface)\n"
        f"I.C.E: Internal Corner Edge, E.C/P.C: Extruded Contour/Portruding Contour, T.H: Through Holes, F: Concave Fillet Surfaces or arches\n"
        f"Use the non-gridded {num_perspective_views} 3D views ONLY to understand the overall shape. The grid numbers should be marked only for the corresponding gridded orthographic images.\n"
        f"Base all cell predictions ONLY on what is visible in the corresponding gridded orthographic image.\n"
        f"For Hollow features mark the boundary cells that contain the hollow feature inside them.\n"
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
    elif prompt_type == 'geo_none':
        return (f"You are given multiple images of a single CAD part along with their camera angles marked in (e: elevation, a: azimuth) format. These angles symbolize the location of the camera with respect to the objects' center in spherical coordinates.\n\n"
        f"The first {num_perspective_views} images show general 3D views of the object (not gridded) with camera situated at various angles respectively: {angles_list_perspective_views}.\n"
        f"The remaining {num_gridded_views} images show orthographic views with camera positions in the list respectively (e: elevation, a: azimuth): {angles_list_gridded_views}.\n"
        f"Each orthographic view has a visible grid with numbered cells.\n\n"
        f"{intro_loading[loading_case]}"
        f"Task:\n"
        f"For EACH gridded orthographic image, identify all grid cells that contain inside them stress critical features.\n"
        f"Use the non-gridded {num_perspective_views} 3D views ONLY to understand the overall shape. The grid numbers should be marked only for the corresponding gridded orthographic images.\n"
        f"Base all cell predictions ONLY on what is visible in the corresponding gridded orthographic image.\n"
        f"Do NOT guess or infer cells that are not clearly visible in that image.\n\n"
        f"Output format (repeat for each gridded image):\n\n"
        f"***Cells: c1, c2, c3 ***\n"
        f"Rules:\n"
        f"- List only cell numbers.\n"
        f"- If no cells apply, write: NONE.\n"
        f"- DO NOT Add explanations \n")
        
info_load_filtering['compression'] = f"Stress critical points are likely to occur across the mid central band of the geometric feature perpendicular to the axis of compression."        
info_load_filtering['bending'] = f"Stress critical points are likely to occur away from the axis of bending (dashed line marked in red) and closer to the outer surface of the geometric feature."
info_load_filtering['torsion'] = f"Stress critical points are likely to occur away from the axis of torsion (dashed line marked in red) and closer to the outer surface of the geometric feature."
info_load_filtering['shear'] = f"Stress critical points are likely to occur closer to the bottom fixed support."
info_load_filtering['bending_compression'] = f"Stress critical points are likely to occur away from the axis of bending (dashed line marked in red) and closer to the outer surface of the geometric feature, as well as across the mid central band of the geometric feature perpendicular to the axis of compression."
info_load_filtering['torsion_compression'] = f"Stress critical points are likely to occur away from the axis of torsion (dashed line marked in red) and closer to the outer surface of the geometric feature, as well as across the mid central band of the geometric feature perpendicular to the axis of compression."


def get_prompt_2(loading_case, prompt_type, perspective_views, gridded_views):
    num_perspective_views = len(perspective_views)
    num_gridded_views = len(gridded_views)

    angles_list_gridded_views, angles_list_perspective_views = [], []
    for view_path in gridded_views:
        el, az = extract_el_az_from_viewpath(view_path)
        angles_list_gridded_views.append((f'e{el}', f'a{az}'))
    for view_path in perspective_views:
        el, az = extract_el_az_from_viewpath(view_path)
        angles_list_perspective_views.append((f'e{el}', f'a{az}'))
    
    if prompt_type in ['geo_max', 'geo_mid']:
        return (f"You are given multiple images of a single CAD part along with their camera angles marked in (e: elevation, a: azimuth) format. These angles symbolize the location of the camera with respect to the objects' center in spherical coordinates.\n\n"
            f"In these {num_gridded_views} orthographic views with angles {angles_list_gridded_views}, filter out cells containing points that are LESS likely to be stress critical. Stress Critical Cells must remain and only the less stress critical cells should be listed out for filtering.\n"
            f"You can see the {loading_case} loading case and the red points at proposed stress critical locations.\n"
            f"Use the non-gridded {num_perspective_views} 3D views ({angles_list_perspective_views}) ONLY to understand the overall shape.\n"
            f"On concave surfaces with stress critical anchor points marked in red, {info_load_filtering[loading_case]}\n"
            f"Task:\n"
            f"Base all cell predictions ONLY on what is visible in the corresponding gridded orthographic image.\n"
            f"Do NOT guess or infer cells that are not clearly visible in that image.\n\n"
            f"Filter out only those cells that contain the predicted red points. Among these cells are those that might not be stress critical. \n\n"
            f"Output format (repeat for each gridded image):\n\n"
            f"***Cells: c1, c2, c3 ***\n"
            f"Rules:\n"
            f"- List only cell numbers.\n"
            f"- If no cells apply, write: NONE.\n"
            f"- Add brief explanations \n")
    elif prompt_type == 'geo_none':
        return (f"You are given multiple images of a single CAD part along with their camera angles marked in (e: elevation, a: azimuth) format. These angles symbolize the location of the camera with respect to the objects' center in spherical coordinates.\n\n"
            f"In these {num_gridded_views} orthographic views with angles {angles_list_gridded_views}, filter out cells containing points that are LESS likely to be stress critical. Stress Critical Cells must remain and only the less stress critical cells should be listed out for filtering.\n"
            f"You can see the {loading_case} loading case and the red points at proposed stress critical locations.\n"
            f"Use the non-gridded {num_perspective_views} 3D views ({angles_list_perspective_views}) ONLY to understand the overall shape.\n"
            f"Task:\n"
            f"Base all cell predictions ONLY on what is visible in the corresponding gridded orthographic image.\n"
            f"Do NOT guess or infer cells that are not clearly visible in that image.\n\n"
            f"Filter out only those cells that contain the predicted red points. Among these cells are those that might not be stress critical. \n\n"
            f"Output format (repeat for each gridded image):\n\n"
            f"***Cells: c1, c2, c3 ***\n"
            f"Rules:\n"
            f"- List only cell numbers.\n"
            f"- If no cells apply, write: NONE.\n"
            f"- Add brief explanations \n")