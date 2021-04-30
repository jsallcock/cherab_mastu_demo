import os

from raysect.primitive import Mesh
from raysect.optical.spectralfunction import ConstantSF
from raysect.optical.material import AbsorbingSurface, Lambert, Roughen
from raysect.optical.library.metal import RoughTungsten, RoughBeryllium, Titanium

try:
    CADMESH_PATH = os.environ['CHERAB_CADMESH']
except KeyError:
    if os.path.isdir('/projects/cadmesh/'):
        CADMESH_PATH = '/projects/cadmesh/'
    else:
        raise ValueError("CHERAB's CAD file path environment variable 'CHERAB_CADMESH' is not set.")

metal_roughness = 0.25
lambertian_absorption = 0.25
METAL = RoughTungsten(metal_roughness)
CARBON = Lambert(ConstantSF(lambertian_absorption))
# METAL = Roughen(Titanium(), metal_roughness)
# CARBON = Roughen(Titanium(), metal_roughness)

def import_mastu_mesh(world, override_material=None, metal_material=None, lambert_material=None):

    LOWER_ELM_COILS = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/magnets/elm_coils/SECTOR_10_LOWER.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/magnets/elm_coils/SECTOR_11_12_LOWER.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/magnets/elm_coils/SECTOR_1_LOWER.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/magnets/elm_coils/SECTOR_2_3_LOWER.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/magnets/elm_coils/SECTOR_4_LOWER.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/magnets/elm_coils/SECTOR_5_6_LOWER.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/magnets/elm_coils/SECTOR_7_LOWER.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/magnets/elm_coils/SECTOR_8_9_LOWER.rsm'), METAL),
    ]

    UPPER_ELM_COILS = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/magnets/elm_coils/SECTOR_11_12_UPPER.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/magnets/elm_coils/SECTOR_2_3_UPPER.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/magnets/elm_coils/SECTOR_5_6_UPPER.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/magnets/elm_coils/SECTOR_8_9_UPPER.rsm'), METAL),
    ]

    ELM_COILS = LOWER_ELM_COILS + UPPER_ELM_COILS

    PF_COILS = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/magnets/pf_coils/P4_LOWER.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/magnets/pf_coils/P4_UPPER.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/magnets/pf_coils/P5_LOWER.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/magnets/pf_coils/P5_UPPER.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/magnets/pf_coils/P6_LOWER.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/magnets/pf_coils/P6_UPPER.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vacuum_vessel/P4_P5_BRACKET1_lower.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vacuum_vessel/P4_P5_BRACKET1_upper.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vacuum_vessel/P4_P5_BRACKET2_lower.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vacuum_vessel/P4_P5_BRACKET2_upper.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vacuum_vessel/P4_P5_BRACKET3_lower.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vacuum_vessel/P4_P5_BRACKET3_upper.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vacuum_vessel/P4_P5_BRACKET4_lower.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vacuum_vessel/P4_P5_BRACKET4_upper.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vacuum_vessel/P4_P5_BRACKET5_lower.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vacuum_vessel/P4_P5_BRACKET5_upper.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vacuum_vessel/P4_P5_BRACKET6_lower.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vacuum_vessel/P4_P5_BRACKET6_upper.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vacuum_vessel/P4_P5_SUPPORT_BRACKETS.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/coil_armour/P5_LOWER_ARMOUR.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/coil_armour/P5_UPPER_ARMOUR.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/coil_armour/P6_LOWER_ARMOUR.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/coil_armour/P6_UPPER_ARMOUR.rsm'), METAL),
    ]

    VACUUM_VESSEL = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vacuum_vessel/OUTER_CYLINDER_VACUUM_VESSEL.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vacuum_vessel/BOTTOM_ENDPLATE.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vacuum_vessel/TOP_ENDPLATE.rsm'), METAL),
    ]

    BEAM_DUMPS = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/beam_dumps/BEAM_DUMP_6M.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/beam_dumps/BEAM_DUMP_8U.rsm'), CARBON),
    ]

    C1_TILE = [(os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/centre_column_armour/C1.rsm'), CARBON)]

    C2_TILES = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/centre_column_armour/C2_lower.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/centre_column_armour/C2_upper.rsm'), CARBON),
    ]

    C3_TILES = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/centre_column_armour/C3_lower.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/centre_column_armour/C3_upper.rsm'), CARBON),
    ]

    C4_TILES = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/centre_column_armour/C4_lower.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/centre_column_armour/C4_upper.rsm'), CARBON),
    ]

    C5_TILES = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/centre_column_armour/C5_lower.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/centre_column_armour/C5_upper.rsm'), CARBON),
    ]

    C6_TILES = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/centre_column_armour/C6_lower.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/centre_column_armour/C6_upper.rsm'), CARBON),
    ]

    CENTRE_COLUMN = C1_TILE + C2_TILES + C3_TILES + C4_TILES + C5_TILES + C6_TILES

    T1_LOWER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/lower/T1_L_01_12.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/lower/T1_L_13_24.rsm'), CARBON),
    ]

    T2_LOWER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/lower/T2_L_01_12.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/lower/T2_L_13_24.rsm'), CARBON),
    ]

    T3_LOWER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/lower/T3_L_01_12.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/lower/T3_L_13_24.rsm'), CARBON),
    ]

    T4_LOWER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/lower/T4_L_01_08.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/lower/T4_L_09_16.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/lower/T4_L_17_24.rsm'), CARBON),
    ]

    T5_LOWER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/lower/T5_L_01_12.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/lower/T5_L_13_24.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/lower/T5_L_25_36.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/lower/T5_L_37_48.rsm'), CARBON),
    ]

    LOWER_DIVERTOR_ARMOUR = T1_LOWER + T2_LOWER + T3_LOWER + T4_LOWER + T5_LOWER

    B1_LOWER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/lower/B1_L_01_12.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/lower/B1_L_13_24.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/lower/B1_L_25_36.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/lower/B1_L_37_48.rsm'), CARBON),
    ]

    B2_LOWER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/lower/B2_L_01_12.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/lower/B2_L_13_24.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/lower/B2_L_25_36.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/lower/B2_L_37_48.rsm'), CARBON),
    ]

    B3_LOWER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/lower/B3_L_01_12.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/lower/B3_L_13_24.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/lower/B3_L_25_36.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/lower/B3_L_37_48.rsm'), CARBON),
    ]

    B4_LOWER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/lower/B4_L_01_12.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/lower/B4_L_13_24.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/lower/B4_L_25_36.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/lower/B4_L_37_48.rsm'), CARBON),
    ]

    N1_LOWER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/lower/N1_L_01_12.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/lower/N1_L_13_24.rsm'), CARBON),
    ]

    N2_LOWER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/lower/N2_L_01_12.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/lower/N2_L_13_24.rsm'), CARBON),
    ]

    LOWER_DIVERTOR_NOSE = B1_LOWER + B2_LOWER + B3_LOWER + B4_LOWER + N1_LOWER + N2_LOWER

    LOWER_GAS_BAFFLE = [(os.path.join(CADMESH_PATH, 'mast/mastu-full/vacuum_vessel/GAS_BAFFLE_LOWER.rsm'), METAL)]

    LOWER_DIVERTOR = LOWER_DIVERTOR_ARMOUR + LOWER_DIVERTOR_NOSE + LOWER_GAS_BAFFLE

    T1_UPPER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/upper/T1_U_01_12.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/upper/T1_U_13_24.rsm'), CARBON),
    ]

    T2_UPPER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/upper/T2_U_01_12.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/upper/T2_U_13_24.rsm'), CARBON),
    ]

    T3_UPPER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/upper/T3_U_01_12.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/upper/T3_U_13_24.rsm'), CARBON),
    ]

    T4_UPPER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/upper/T4_U_01_08.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/upper/T4_U_09_16.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/upper/T4_U_17_24.rsm'), CARBON),
    ]

    T5_UPPER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/upper/T5_U_01_12.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/upper/T5_U_13_24.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/upper/T5_U_25_36.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/divertor_armour/upper/T5_U_37_48.rsm'), CARBON),
    ]

    UPPER_DIVERTOR_ARMOUR = T1_UPPER + T2_UPPER + T3_UPPER + T4_UPPER + T5_UPPER

    B1_UPPER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/upper/B1_U_01_12.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/upper/B1_U_13_24.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/upper/B1_U_25_36.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/upper/B1_U_37_48.rsm'), CARBON),
    ]

    B2_UPPER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/upper/B2_U_01_12.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/upper/B2_U_13_24.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/upper/B2_U_25_36.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/upper/B2_U_37_48.rsm'), CARBON),
    ]

    B3_UPPER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/upper/B3_U_01_12.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/upper/B3_U_13_24.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/upper/B3_U_25_36.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/upper/B3_U_37_48.rsm'), CARBON),
    ]

    B4_UPPER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/upper/B4_U_01_12.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/upper/B4_U_13_24.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/upper/B4_U_25_36.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/upper/B4_U_37_48.rsm'), CARBON),
    ]

    N1_UPPER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/upper/N1_U_01_12.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/upper/N1_U_13_24.rsm'), CARBON),
    ]

    N2_UPPER = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/upper/N2_U_01_12.rsm'), CARBON),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/vessel_armour/throat_armour/upper/N2_U_13_24.rsm'), CARBON),
    ]

    UPPER_DIVERTOR_NOSE = B1_UPPER + B2_UPPER + B3_UPPER + B4_UPPER + N1_UPPER + N2_UPPER

    UPPER_GAS_BAFFLE = [(os.path.join(CADMESH_PATH, 'mast/mastu-full/vacuum_vessel/GAS_BAFFLE_UPPER.rsm'), METAL)]

    UPPER_DIVERTOR = UPPER_DIVERTOR_ARMOUR + UPPER_DIVERTOR_NOSE + UPPER_GAS_BAFFLE

    MASTU_FULL_MESH = ELM_COILS + PF_COILS + VACUUM_VESSEL + BEAM_DUMPS + CENTRE_COLUMN + LOWER_DIVERTOR + UPPER_DIVERTOR

    SXD_BOLOMETERS = [
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/diagnostics/DIVERTOR_horizontal_bolometer.rsm'), METAL),
        (os.path.join(CADMESH_PATH, 'mast/mastu-full/diagnostics/DIVERTOR_verticall_bolometer.rsm'), METAL),
    ]

    for mesh_item in MASTU_FULL_MESH:

        mesh_path, default_material = mesh_item

        if override_material:
            material = override_material
        elif metal_material and isinstance(default_material, RoughTungsten):
            material = metal_material
        elif lambert_material and isinstance(default_material, Lambert):
            material = lambert_material
        else:
            material = default_material

        print("importing {}  ...".format(os.path.split(mesh_path)[1]))
        directory, filename = os.path.split(mesh_path)
        mesh_name, ext = filename.split('.')
        Mesh.from_file(mesh_path, parent=world, material=material, name=mesh_name)

    return world
