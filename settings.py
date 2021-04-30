from raysect.core import Vector3D, Point3D
from cherab.core.atomic.elements import deuterium, carbon

view = {'HL-06 (MWI)': {'pupil_point': Point3D(0.709, -2.002, -1.523, ),
                        'view_vector': Vector3D(-0.7464, 0.6307, -0.2125, ),
                        },
        'HL-02 (CI)': {'pupil_point': Point3D(1.399, 1.649, -1.509),
                       'view_vector': Vector3D(-0.21481961, -0.95194258, -0.21830681),
                       },
        'midplane': {'pupil_point': Point3D(0, 1.9, 0, ),
                     'view_vector': Vector3D(0, -1, 0, ),
                     },
        }

solps = [  # MAST-U SXD, Omkar, ref. no. from Kevin.
    121829,  # med dens
    121819,  # low dens
    121837,  # hi dens

    69636,  # sxd1h
    69566,
]

line = {'D_gamma': {'element': deuterium,
                    'charge': 0,
                    'transition': (5, 2,),
                    },
        'D_beta': {'element': deuterium, 'charge': 0,
                   'transition': (4, 2,),
                   },
        'D_delta': {'element': deuterium,
                    'charge': 0,
                    'transition': (6, 2,),
                    },
        }

# FLIR Blackfly S format is actually (2448, 2048) but this is neater for downsampling
# Similarly the Ximea camera is actually (2064, 1544, )
camera = {'FLIR_Blackfly': {'sensor_format': (2550, 2050,),
                            'pixel_size': 3.45e-6,
                            'bit_depth': 12,
                            'qe': 0.35,
                            'epercount': 0.18,
                            'cam_noise': 2.38,
                            },
          'FLIR_Blackfly_reduced': {'sensor_format': (2550, 1400,),
                             'pixel_size': 3.45e-6,
                             'bit_depth': 12,
                             'qe': 0.35,
                             'epercount': 0.18,
                             'cam_noise': 2.38,
                             },
          'Ximea_MX031xG-SY-X2G2-Fx': {'sensor_format': (2060, 1540,),
                                       'pixel_size': 3.45e-6,
                                       'bit_depth': 12,
                                       'qe': 0.35,
                                       'epercount': 1,
                                       'cam_noise': 2.32,
                                       },
          'test_cam': {'sensor_format': (310, 250),
                       'pixel_size': 3.45e-6 * 8,
                       'bit_depth': 12,
                       'qe': 0.35,
                       'epercount': 0.46,
                       'cam_noise': 2.5,
                       },
          }

# this is just a list of lens focal lengths
optics = {'MWI_CI': [20e-3, 102.2e-3, 50e-3, ],
          'MAST_CI_mod': [17e-3, 150e-3, 105e-3, ],
          'MAST_CI_wide_view': [17e-3, 105e-3, 150e-3, ],
          }