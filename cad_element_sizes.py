"""Element size lookup extracted from compute_zz_pos.log.

Generated on 2026-05-02. Values are deduplicated by CAD ID and
verified to be independent of loading case in the source log.
"""

CAD_ELEMENT_SIZES = {
    '00200002': {'h_max': 5.0092, 'h_min': 1.0018, 'h_fine': 0.5009},
    '00200005': {'h_max': 7.1092, 'h_min': 1.4218, 'h_fine': 0.7109},
    '00200008': {'h_max': 4.1140, 'h_min': 0.8228, 'h_fine': 0.4114},
    '00200022': {'h_max': 4.5707, 'h_min': 0.9141, 'h_fine': 0.4571},
    '00200030': {'h_max': 14.4687, 'h_min': 2.8937, 'h_fine': 1.4469},
    '00200037': {'h_max': 3.6695, 'h_min': 0.7339, 'h_fine': 0.3669},
    '00200039': {'h_max': 10.9612, 'h_min': 2.1922, 'h_fine': 1.0961},
    '00200050': {'h_max': 4.7898, 'h_min': 0.9580, 'h_fine': 0.4790},
    '00200069': {'h_max': 2.7703, 'h_min': 0.5541, 'h_fine': 0.2770},
    '00200089': {'h_max': 12.5427, 'h_min': 2.5085, 'h_fine': 1.2543},
    '00200090': {'h_max': 9.4329, 'h_min': 1.8866, 'h_fine': 0.9433},
    '00210005': {'h_max': 10.7187, 'h_min': 2.1437, 'h_fine': 1.0719},
    '00210018': {'h_max': 12.3144, 'h_min': 2.4629, 'h_fine': 1.2314},
    '00210021': {'h_max': 6.5333, 'h_min': 1.3067, 'h_fine': 0.6533},
    '00210058': {'h_max': 5.6326, 'h_min': 1.1265, 'h_fine': 0.5633},
    '00210070': {'h_max': 4.2060, 'h_min': 0.8412, 'h_fine': 0.4206},
    '00210076': {'h_max': 0.7421, 'h_min': 0.1484, 'h_fine': 0.0742},
    '00210090': {'h_max': 3.5248, 'h_min': 0.7050, 'h_fine': 0.3525},
    '00210097': {'h_max': 5.1402, 'h_min': 1.0280, 'h_fine': 0.5140},
    '00220004': {'h_max': 5.6364, 'h_min': 1.1273, 'h_fine': 0.5636},
    '00220071': {'h_max': 5.8257, 'h_min': 1.1651, 'h_fine': 0.5826},
    '00220074': {'h_max': 9.5565, 'h_min': 1.9113, 'h_fine': 0.9556},
    '00230003': {'h_max': 4.1929, 'h_min': 0.8386, 'h_fine': 0.4193},
    '00230017': {'h_max': 4.7575, 'h_min': 0.9515, 'h_fine': 0.4758},
    '00510024': {'h_max': 19.6173, 'h_min': 3.9235, 'h_fine': 1.9617},
    '00520006': {'h_max': 5.8120, 'h_min': 1.1624, 'h_fine': 0.5812},
    '00520017': {'h_max': 0.9197, 'h_min': 0.1839, 'h_fine': 0.0920},
    '00520044': {'h_max': 2.9177, 'h_min': 0.5835, 'h_fine': 0.2918},
    '00520092': {'h_max': 11.4957, 'h_min': 2.2991, 'h_fine': 1.1496},
    '00530042': {'h_max': 6.2104, 'h_min': 1.2421, 'h_fine': 0.6210},
    '00530061': {'h_max': 11.3218, 'h_min': 2.2644, 'h_fine': 1.1322},
    '00530079': {'h_max': 7.4797, 'h_min': 1.4959, 'h_fine': 0.7480},
    '00540016': {'h_max': 6.3697, 'h_min': 1.2739, 'h_fine': 0.6370},
    '00550014': {'h_max': 5.6739, 'h_min': 1.1348, 'h_fine': 0.5674},
    'Electrical_Parts_Servos_SG-90_SG90-1-arm-horn': {'h_max': 1.1193, 'h_min': 0.2239, 'h_fine': 0.1119},
    'Electrical_Parts_Servos_SG-90_SG90-4-arms-horn': {'h_max': 1.3002, 'h_min': 0.2600, 'h_fine': 0.1300},
    'Electrical_Parts_Servos_SG-90_Servo-sg90': {'h_max': 3.2666, 'h_min': 0.6533, 'h_fine': 0.3267},
    'Mechanical_Parts_Mountings_SC8UU_SC8UU': {'h_max': 2.0246, 'h_min': 0.4049, 'h_fine': 0.2025},
    'Mechanical_Parts_Mountings_SHF08_SHF08': {'h_max': 3.1268, 'h_min': 0.6254, 'h_fine': 0.3127},
    'Mechanical_Parts_Mountings_SK08_SK08-SK08': {'h_max': 3.5487, 'h_min': 0.7097, 'h_fine': 0.3549},
}

def get_element_sizes(cad_id: str):
    return CAD_ELEMENT_SIZES[cad_id]
