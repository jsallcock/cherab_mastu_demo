#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'settings'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'mastu_cad_mesh'))
import settings
from mastu_cad_mesh import import_mastu_mesh
import matplotlib.pyplot as plt

import numpy as np
from numba import vectorize, float64, complex128
import xarray as xr
from scipy.constants import c, h
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
# plt.ion()

from raysect.core import Vector3D, Point3D, translate, rotate, rotate_x, rotate_y, rotate_z, rotate_vector
from raysect.core.ray import Ray as CoreRay
from raysect.optical import World, ConstantSF, Spectrum
from raysect.optical.observer import PinholeCamera
from raysect.optical.observer import RGBPipeline2D, SpectralPowerPipeline2D, PowerPipeline2D, SpectralRadiancePipeline2D

from cherab.core.atomic import Line
from cherab.core.atomic.elements import deuterium, carbon
from cherab.core.model import ExcitationLine, RecombinationLine, Bremsstrahlung
from cherab.solps import load_solps_from_mdsplus
from cherab.openadas import OpenADAS
from cherab.core.model.lineshape import GaussianLine, LineShapeModel, StarkBroadenedLine

import gc
gc.enable()

mds_server = 'solps-mdsplus.aug.ipp.mpg.de:8001'
path_saved_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_data')

a = 5
class CherabMASTU(object):
    """
    demo class for modelling Cherab images of MAST-U
    """

    def __init__(self, solps_ref_no, view, line, camera, optics='MAST_CI_mod', cherab_down_sample=50, verbose=True,
                 display=False, overwrite=False):
        """
        :param solps_ref_no: int
        :param view: str
        :param line: str
        :param camera: str
        :param optics: str
        :param cherab_down_sample: int downsample the image by ths factor in both dimensions, for speed / testing.
        :param verbose: bool
        :param display: bool
        :param overwrite: bool
        """

        self.world = World()
        self.solps_ref_no = solps_ref_no
        self.line = line
        self.view = view
        self.camera = camera
        self.optics = optics
        self.cherab_down_sample = cherab_down_sample
        self.verbose = verbose
        self.display = display
        self.overwrite = overwrite

        self.radiance = NotImplemented
        self.spectral_radiance = NotImplemented

        self.pupil_point = settings.view[self.view]['pupil_point']
        self.view_vector = settings.view[self.view]['view_vector']
        self.sensor_format = settings.camera[self.camera]['sensor_format']
        for s in self.sensor_format:
            assert s % cherab_down_sample == 0
        self.sensor_format_ds = tuple((np.array(self.sensor_format) / cherab_down_sample).astype(np.int))
        self.pixel_size = settings.camera[self.camera]['pixel_size']
        self.pixel_size_ds = self.pixel_size * cherab_down_sample

        self.x, self.y, = self.calc_pixel_position(self.pixel_size, self.sensor_format)
        self.x_pixel = self.x.x_pixel
        self.y_pixel = self.y.y_pixel

        self.x_ds = self.x.isel(x=slice(0, self.sensor_format[0], cherab_down_sample, ))
        self.y_ds = self.y.isel(y=slice(0, self.sensor_format[1], cherab_down_sample, ))
        self.x_pixel_ds = self.x_ds.x_pixel
        self.y_pixel_ds = self.y_ds.y_pixel

        self.chunks = {'x': 800, 'y': 800}

        # calculate field of view (FOV) (horizontal) using sensor geometry and lens focal lengths
        f_1, f_2, f_3 = settings.optics[self.optics]
        sensor_dim = self.sensor_format[0] * self.pixel_size
        self.fov = 2 * np.arctan((sensor_dim / 2) * f_2 / (f_3 * f_1)) * 180 / np.pi  # deg
#
        # file and directory paths
        self.dir_name = str(solps_ref_no) + '_' + view + '_' + line + '_' + camera
        self.dir_path = os.path.join(path_saved_data, self.dir_name)
        fname = 'spec_power.nc'
        fname_ds = 'spec_power_ds.nc'
        self.fpath_ds = os.path.join(self.dir_path, fname_ds, )
        self.fpath = os.path.join(self.dir_path, fname, )

        # load SOLPS plasma and set emission line model
        self.sim = load_solps_from_mdsplus(mds_server, self.solps_ref_no)
        self.plasma = self.sim.create_plasma(parent=self.world)
        self.plasma.atomic_data = OpenADAS(permit_extrapolation=True)

        emission_line = settings.line[self.line]
        element, charge, transition = [emission_line[s] for s in ['element', 'charge', 'transition', ]]
        line_obj = Line(element, charge, transition, )
        self._line_excit = ExcitationLine(line_obj, lineshape=StarkBroadenedLine)
        self._line_recom = RecombinationLine(line_obj, lineshape=StarkBroadenedLine)
        self.plasma.models = [Bremsstrahlung(), self._line_excit, self._line_recom]
        wl_0_nm = self._line_excit.atomic_data.wavelength(element, charge, transition, )
        self.wl_0 = wl_0_nm * 1e-9  # [m]
        self.wl_min_nm, self.wl_max_nm = wl_0_nm - 0.2, wl_0_nm + 0.2  # [m]

        # ugly, but I want to convert the density units to 10^{20} m^{-3}
        def get_ne(x, y, z, ):
            return 1e-20 * self.plasma.electron_distribution.density(x, y, z, )

        def get_ni(x, y, z, ):
            return 1e-20 * self.plasma.composition.get(deuterium, 0).distribution.density(x, y, z, )

        def get_b_field_mag(x, y, z, ):
            """
            magnitude of the magnetic field at x, y, z
            :param x:
            :param y:
            :param z:
            :return:
            """
            return self.plasma.b_field(x, y, z, ).length

        def get_emiss_excit(x, y, z, ):
            return self._line_excit.emission(Point3D(x, y, z, ), Vector3D(1, 1, 1), Spectrum(380, 700, 100)).total()

        def get_emiss_recom(x, y, z, ):
            return self._line_recom.emission(Point3D(x, y, z, ), Vector3D(1, 1, 1), Spectrum(380, 700, 100)).total()

        self.valid_param_get_fns = {'ne': get_ne,
                                    'ni': get_ni,
                                    'te': self.plasma.electron_distribution.effective_temperature,
                                    'ti': self.plasma.composition.get(deuterium, 0).distribution.effective_temperature,
                                    'tn': self.plasma.composition.get(deuterium, 1).distribution.effective_temperature,
                                    'b_field_mag': get_b_field_mag,
                                    'emiss_excit': get_emiss_excit,
                                    'emiss_recom': get_emiss_recom,
                                    }
        self.valid_params = list(self.valid_param_get_fns.keys())

        # load / make the cherab image
        if os.path.isdir(self.dir_path) and self.overwrite is False:
            if os.path.isfile(self.fpath) and os.path.isfile(self.fpath_ds):
                pass
        else:
            if not os.path.isdir(self.dir_path):
                os.mkdir(self.dir_path)
                self.make_cherab_image()

        ds, ds_ds = self.load_cherab_image()
        self.spectral_radiance = ds['spectral_radiance']
        self.radiance = self.spectral_radiance.integrate(dim='wavelength')
        self.radiance = xr.where(xr.ufuncs.isnan(self.radiance), 0, self.radiance, )

        self.spectral_radiance_ds = ds_ds['spectral_radiance_ds']
        self.radiance_ds = ds_ds['radiance_ds']
        self.view_vectors = ds_ds['view_vectors_ds']
        self.ray_lengths = ds_ds['ray_lengths_ds']
        ds.close()

        # self.mask_ds = self.make_mask_ds()

    def load_cherab_image(self):
        ds = xr.open_dataset(self.fpath, chunks=self.chunks)
        ds_ds = xr.load_dataset(self.fpath_ds, )
        return ds, ds_ds,

    def make_cherab_image(self):
        """
        run cherab to generate the synthetic spectral cube
        :return:
        """
        if self.radiance is not NotImplemented:
            self.radiance.close()
        if self.spectral_radiance is not NotImplemented:
            self.spectral_radiance.close()

        import_mastu_mesh(self.world, )

        # first, define camera, calculate view vectors and calculate ray lengths
        pipeline_spectral = SpectralPowerPipeline2D()
        pipeline_spectral_rad = SpectralRadiancePipeline2D()
        pipelines = [pipeline_spectral, pipeline_spectral_rad, ]
        camera = PinholeCamera(self.sensor_format_ds, fov=self.fov, pipelines=pipelines, parent=self.world)

        # orient and position the camera
        init_view_vector, init_up_vector = Vector3D(0, 0, 1), Vector3D(0, 1, 0)
        axle_1 = init_view_vector.cross(self.view_vector)
        angle = init_view_vector.angle(self.view_vector)
        t_1 = rotate_vector(angle, axle_1)

        final_up_vector = rotate_vector(-90, axle_1) * self.view_vector
        intermediate_up_vector = t_1 * init_up_vector
        angle_between = intermediate_up_vector.angle(final_up_vector)
        t_2 = rotate_vector(-angle_between, self.view_vector)

        camera.transform = translate(self.pupil_point[0],
                                     self.pupil_point[1],
                                     self.pupil_point[2], ) * t_2 * t_1

        vector_xyz = np.arange(3)
        vector_xyz = xr.DataArray(vector_xyz, coords=(vector_xyz, ), dims=('vector_xyz',), name='vector_xyz', )

        # calculating the pixel view directions
        view_vectors = xr.combine_nested(
            [xr.zeros_like(self.x_pixel_ds + self.y_pixel_ds) + self.view_vector[i] for i in [0, 1, 2, ]],
            concat_dim=(vector_xyz,), )
        view_vectors = view_vectors.rename('view_vectors')

        def v3d2da(v3d):
            """
            raysect Vector3D to xarray DataArray

            :param v3d:
            :return:
            """
            da = np.array([v3d.x, v3d.y, v3d.z, ])
            da = xr.DataArray(da, coords=(np.arange(3),), dims=('vector_xyz',), )
            return da

        # basis unit vectors defining camera view -- v_z is forward and v_y is up
        v_y = final_up_vector.normalise()
        v_x = self.view_vector.cross(v_y).normalise()
        v_z = self.view_vector.normalise()
        v_x, v_y, v_z = [v3d2da(i) for i in [v_x, v_y, v_z, ]]

        # FOV defines the widest view, with pixels defined as square.
        sensor_aspect = self.sensor_format[1] / self.sensor_format[0]
        if sensor_aspect > 1:
            fov_v = self.fov
            fov_h = self.fov / sensor_aspect
        elif sensor_aspect == 1:
            fov_v = fov_h = self.fov
        elif sensor_aspect < 1:
            fov_h = self.fov
            fov_v = self.fov * sensor_aspect
        else:
            raise Exception()

        pixel_projection = 2 * np.tan(fov_h * np.pi / 360) / self.sensor_format[0]
        view_vectors = view_vectors + (v_x * (self.x_pixel_ds - self.sensor_format[0] / 2 + 0.5) * pixel_projection) + \
                       (v_y * (self.y_pixel_ds - self.sensor_format[1] / 2 + 0.5) * pixel_projection)

        if self.verbose:
            print('--status: calculating ray lengths')
        # TODO there has to be a better way of doing this?!
        ray_lengths = xr.DataArray(np.zeros(self.sensor_format_ds), dims=('x', 'y', ), coords=(self.x_ds, self.y_ds, ))
        for idx_x, x_pixel in enumerate(self.x_pixel_ds.values):
            if self.verbose and idx_x % 10 == 0:
                print('x =', str(x_pixel))
            for idx_y, y_pixel in enumerate(self.y_pixel_ds.values):
                direction = Vector3D(*list(view_vectors.isel(x=idx_x, y=idx_y, ).values))

                intersections = []
                for p in self.world.primitives:
                    intersection = p.hit(CoreRay(self.pupil_point, direction, ))
                    if intersection is not None:
                        intersections.append(intersection)

                # find the intersection corresponding to the shortest ray length
                no_intersections = len(intersections)
                if no_intersections == 0:
                    ray_lengths.values[idx_x, idx_y] = 3
                else:
                    ray_lengths.values[idx_x, idx_y] = min([i.ray_distance for i in intersections if i.primitive.name != 'Plasma Geometry'])

        camera.spectral_bins = 40
        camera.pixel_samples = 10
        camera.min_wavelength = self.wl_min_nm
        camera.max_wavelength = self.wl_max_nm
        camera.quiet = not self.verbose
        camera.observe()

        # output to netCDF via xarray
        wl = pipeline_spectral.wavelengths
        wl = xr.DataArray(wl, coords=(wl, ), dims=('wavelength', )) * 1e-9  # ( m )
        spec_power_ds = pipeline_spectral.frame.mean * 1e9  # converting units from (W/nm) --> (W/m)
        spec_radiance_ds = pipeline_spectral_rad.frame.mean * 1e9
        coords = (self.x_ds, self.y_ds, wl, )
        dims = ('x', 'y', 'wavelength', )
        name = 'spec_power'
        attrs = {'units': 'W/m^2/str/m'}
        spec_power_ds = xr.DataArray(np.flip(spec_power_ds, axis=1), coords=coords, dims=dims, name=name, attrs=attrs, )
        spec_radiance_ds = xr.DataArray(np.flip(spec_radiance_ds, axis=1, ), coords=coords, dims=dims, name=name, attrs=attrs, )

        # calculate the centre-of-mass wavelength
        radiance_ds = spec_power_ds.integrate(dim='wavelength').assign_attrs({'units': 'W/m^2/str', })

        ds_ds = xr.Dataset({'spectral_radiance_ds': spec_radiance_ds,
                            'radiance_ds': radiance_ds,
                            'view_vectors_ds': view_vectors,
                            'ray_lengths_ds': ray_lengths
                            })

        x_p_y = self.x + self.y
        spec_power = spec_power_ds.interp_like(x_p_y) / self.cherab_down_sample  # to conserve power
        ds = xr.Dataset({'spectral_radiance': spec_power, })
        ds_ds.to_netcdf(self.fpath_ds, mode='w', )
        ds.to_netcdf(self.fpath, mode='w', )

    def plot_line_of_sight(self, x_pixel, y_pixel, ax, **kwargs):
        """

        :param x_pixel:
        :param y_pixel:
        :param ax:
        :return:
        """

        direction = Vector3D(*list(self.view_vectors.isel(x=x_pixel, y=y_pixel, ).values)).normalise()
        ray_length = float(self.ray_lengths.isel(x=x_pixel, y=y_pixel, ).values)

        n_steps = 50
        increment = (ray_length / n_steps) * direction
        rs, zs = np.zeros(n_steps), np.zeros(n_steps)
        point_i = self.pupil_point  # initialise at pupil
        for i in range(n_steps):
            point_i += increment
            rs[i], zs[i], _ = cart2cyl(*[point_i[j] for j in range(3)])
        ax.plot(rs, zs, **kwargs)

    def calc_pixel_position(self, pixel_size, sensor_format, ):
        """
        Calculate pixel positions (in m) on the camera's sensor plane (the x-y plane).

        The origin of the x-y coordinate system is the centre of the sensor. Pixel positions correspond to the pixel
        centres. If x_pixel and y_pixel are specified then only returns the position of that pixel.

        :param float pixel_size: in m
        :param tuple sensor_format: (num_pix_x, num_pix_y)
        :return:
        """

        centre_pos = pixel_size * np.array(sensor_format) / 2
        x = (np.arange(sensor_format[0]) + 0.5) * pixel_size - centre_pos[0]
        y = (np.arange(sensor_format[1]) + 0.5) * pixel_size - centre_pos[1]
        x = xr.DataArray(x, dims=('x',), coords=(x,), )
        y = xr.DataArray(y, dims=('y',), coords=(y,), )

        x_pixel_coord = xr.DataArray(np.arange(sensor_format[0], ), dims=('x',), coords=(x,), )
        y_pixel_coord = xr.DataArray(np.arange(sensor_format[1], ), dims=('y',), coords=(y,), )
        x = x.assign_coords({'x_pixel': ('x', x_pixel_coord), }, )
        y = y.assign_coords({'y_pixel': ('y', y_pixel_coord), }, )

        return x, y


def cart2cyl(x, y, z):
    """
    Cartesian to cylindrical coordinates

    """
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return r, z, theta


if __name__ == '__main__':
    # QUICK DEMO -- heavily downsampled!
    args = {
        'solps_ref_no': 121829,
        'view': 'HL-06 (MWI)',
        'line': 'D_gamma',
        'camera': 'FLIR_Blackfly',
        'optics': 'MWI_CI'
    }
    cmu = CherabMASTU(**args, cherab_down_sample=50)
    cmu.radiance.plot(x='x', y='y', vmin=0, )
    plt.show()