import caw
import capt
import numpy
from astropy.io import fits
from esoTurbProfile import esoTurbProfile
from matplotlib import pyplot; pyplot.ion()
import capt.misc_functions.matplotlib_format


def get_turb_profile(turb_conf, air_mass, tas, pix_arc, shwfs_centroids, input_matrix=False):
    turb_conf = capt.turbulence_profiler(turb_conf)
    turb_results = turb_conf.perform_turbulence_profiling(air_mass, tas, pix_arc, shwfs_centroids)
    return turb_results


def get_wind_profile(turb_conf, wind_conf, frame_rate, frame_count, air_mass, tas, pix_arc, shwfs_centroids):
    turb_results = get_turb_profile(turb_conf, air_mass, tas, pix_arc, shwfs_centroids)
    wind_conf = caw.wind_profiler(turb_results, wind_conf)
    wind_results = wind_conf.perform_wind_profiling(frame_rate, frame_count=frame_count)
    return wind_results


if __name__ == '__main__':

    """FIT WIND PROFILE"""
    air_mass = 1.
    frame_rate = 150.
    frame_count = numpy.arange(10000)
    pix_arc = numpy.nan
    shwfs_centroids = fits.getdata('test_fits/canary_noNoise_it10k_nl3_h0a10a20km_r00p1_L025_ws10a15a20_wd260a80a350_infScrn_wss448_gsPos0cn40a0c0a30c0.fits')
    tas = numpy.array([[0., -40.], [0., 0.], [30., 0.]])
    turb_conf = capt.configuration('../conf/turb_conf.yaml')
    wind_conf = caw.configuration('../conf/wind_conf.yaml')
    
    turb_results = get_turb_profile(turb_conf, air_mass, tas, pix_arc, shwfs_centroids)
    wind_results = get_wind_profile(turb_conf, wind_conf, frame_rate, frame_count, air_mass, tas, pix_arc, shwfs_centroids)

    # pyplot.figure('analytical fit')
    # pyplot.imshow(results.wind_fit.covMapOffset)
    # pyplot.figure('measured')
    # pyplot.imshow(results.roi_offsets[0])