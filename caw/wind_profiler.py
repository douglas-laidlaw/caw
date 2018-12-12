import yaml
import time
import capt
import numpy
import itertools
from scipy.misc import comb
from matplotlib import pyplot; pyplot.ion()
from capt.misc_functions.cross_cov import cross_cov
from capt.misc_functions.calc_Cn2_r0 import calc_r0
from capt.roi_functions.roi_from_map import roi_from_map
from capt.misc_functions.mapping_matrix import get_mappingMatrix
from capt.misc_functions.transform_matrix import transform_matrix
from capt.map_functions.covMap_fromMatrix import covMap_fromMatrix
from capt.roi_functions.roi_referenceArrays import roi_referenceArrays
from capt.roi_functions.roi_zeroSep_locations import roi_zeroSep_locations
from caw.fitting_functions.wind_fitting_algorithm import fitting_parameters
from capt.roi_functions.calculate_roi_covariance import calculate_roi_covariance




class configuration(object):
    """First function to be called. The specified configuration file is imported and 
    exceptions called if there are variable inconsistencies."""

    def __init__(self, config_file):
        """Reads in configuration file and checks for variable inconsistencies
        
        Parameters:
            config_file (yaml): configuration file for turbulence profiling."""

        self.loadYaml(config_file)

        fit_params = self.configDict["FITTING ALGORITHM"]
        delta_xSep = numpy.array(fit_params["delta_xSep"])
        delta_ySep = numpy.array(fit_params["delta_ySep"])
        
        #raise exception if specified layer parameters do not match.
        if len(delta_xSep)!=len(delta_ySep):
            raise Exception('Check lengths of wind parameters.')


    def loadYaml(self, yaml_file):
        with open(yaml_file) as _file:
            self.configDict = yaml.load(_file)






class wind_profiler(object):
    """Master node for performing turbulence profiling
    NOTE: LGS configurations have not been fully tested."""

    def __init__(self, turb_results, wind_config_file):
        """Fixes variables from config_file that are to be used to complete turbulence profiling.

        Parameters:
        config_file (dict): fitting specifications set in imported yaml configuration file."""


        self.turb_results = turb_results
        self.tel_diam = self.turb_results.tel_diam
        self.n_wfs = self.turb_results.n_wfs
        self.n_subap = self.turb_results.n_subap
        self.n_subap_from_pupilMask = turb_results.n_subap_from_pupilMask
        self.nx_subap = self.turb_results.nx_subap
        self.pupil_mask = self.turb_results.pupil_mask
        self.shwfs_centroids = self.turb_results.shwfs_centroids
        self.gs_pos = self.turb_results.gs_pos
        self.n_layer = self.turb_results.n_layer
        self.observable_bins = self.turb_results.observable_bins

        self.L0 = self.turb_results.L0
        self.tt_track = self.turb_results.tt_track
        self.lgs_track = self.turb_results.lgs_track
        self.combs = self.turb_results.combs
        self.selector = self.turb_results.selector
        self.shwfs_shift = self.turb_results.shwfs_shift
        self.shwfs_rot = self.turb_results.shwfs_rot
    
        target_conf = wind_config_file.configDict["OFFSET TARGET ARRAY"]
        self.input_shwfs_centroids = target_conf["input_shwfs_centroids"]
        self.roi_via_matrix = target_conf["roi_via_matrix"]
        self.zeroSep_cov = target_conf["zeroSep_cov"]
        self.input_frame_count = target_conf["input_frame_count"]
        self.num_offsets = target_conf["num_offsets"]
        self.offset_step = target_conf["offset_step"]
        self.temporal_step = numpy.round(float(self.offset_step)/float(self.num_offsets)).astype('int')
        self.include_temp0 = target_conf["include_temp0"]
        self.minus_negativeTemp = target_conf["minus_negativeTemp"]
        self.separate_pos_neg_offsets = target_conf["separate_pos_neg_offsets"]
        self.wind_roi_belowGround = target_conf["roi_belowGround"]
        self.wind_roi_envelope = target_conf["roi_envelope"]
        self.wind_map_axis = target_conf["map_axis"]
        self.wind_mapping_type = target_conf["mapping_type"]

        fit_params = wind_config_file.configDict["FITTING ALGORITHM"]
        self.delta_xSep = numpy.array(fit_params["delta_xSep"]).astype('float64')
        self.delta_ySep = numpy.array(fit_params["delta_ySep"]).astype('float64')
        if len(self.delta_xSep)!=self.n_layer:
            raise Exception('Check length of wind parameters with n_layer.')



        fit_conf = wind_config_file.configDict["FITTING ALGORITHM"]
        self.fitting_type = fit_conf["type"]
        self.fit_layer_alt = fit_conf["fit_layer_alt"]
        self.fit_deltaXYseps = fit_conf["fit_deltaXYseps"]
        self.print_fiting = fit_conf["print_fitting"]

        self.onesMat, self.wfsMat_1, self.wfsMat_2, self.allMapPos, self.selector, self.xy_separations = roi_referenceArrays(
                self.pupil_mask, self.gs_pos, self.tel_diam, self.wind_roi_belowGround, self.wind_roi_envelope)
        
        if self.zeroSep_cov==False:
            roi_width = (2*self.wind_roi_envelope) + 1
            roi_length = self.pupil_mask.shape[0] + self.wind_roi_belowGround
            self.zeroSep_locations = roi_zeroSep_locations(self.combs, roi_width, 
                roi_length, self.wind_map_axis, self.wind_roi_belowGround)
        else:
            self.zeroSep_locations = numpy.nan


        if self.minus_negativeTemp==True:
            self.mult_neg_offset = -1.
        else:
            self.mult_neg_offset = 1.


        if wind_config_file.configDict["FITTING ALGORITHM"]["type"]=='direct':
            self.fit_method = 'Direct Fit'
            self.direct_wind = True
            self.l3s_wind = False
            self.l3s1_matrix = numpy.nan


        if wind_config_file.configDict["FITTING ALGORITHM"]["type"]=='l3s':
            self.fit_method = 'L3S Fit'
            self.l3s_wind = True
            self.direct_wind = False
            if self.turb_results.l3s_fit==True:
                self.l3s1_matrix = self.turb_results.l3s1_matrix
            else:
                self.l3s1_matrix = transform_matrix(self.n_subap_from_pupilMask, self.n_wfs)


        if self.turb_results.covariance_map==True or self.turb_results.covariance_map_roi==True:
            self.mm = self.turb_results.mm
            self.mmc = self.turb_results.mmc
            self.md = self.turb_results.md
        else:
            matrix_region_ones = numpy.ones((self.n_subap_from_pupilMask[0], self.n_subap_from_pupilMask[0]))
            self.mm, self.mmc, self.md = get_mappingMatrix(self.pupil_mask, matrix_region_ones)


        print('\n'+'###########################################################','\n')
        print('############ WIND PROFILING PARAMETERS SECURE #############')







    def perform_wind_profiling(self, frame_rate, frame_count=False, shwfs_cents=False):

        if self.input_shwfs_centroids==True:
            self.shwfs_centroids = shwfs_cents

        if self.input_frame_count==True:
            if frame_count.shape[0]!=self.shwfs_centroids.shape[0]:
                raise Exception('Input frame count not the same length as SHWFS iterations.')
            self.shwfs_centroids = self.shwfs_centroids[numpy.argsort(frame_count)]
            frame_count = frame_count[numpy.argsort(frame_count)]
            nearest = numpy.abs(frame_count-self.temporal_step).argmin()
            self.temporal_step = frame_count[nearest]

        start_calc = time.time()
        self.roi_offsets = self.temporal_offset_roi()
        self.time_calc_offsets = time.time() - start_calc

        # wind profiling configuration
        wind_params = fitting_parameters(self.turb_results, self.fit_method, self.roi_offsets, 
            frame_rate, self.num_offsets, self.offset_step, self.wind_roi_belowGround, 
            self.wind_roi_envelope, self.wind_map_axis, self.zeroSep_cov, self.zeroSep_locations, 
            self.include_temp0, self.mult_neg_offset, self.separate_pos_neg_offsets, 
            self.print_fiting)
        
        start_fit = time.time()
        self.wind_fit = wind_params.fit_roi_offsets(self.delta_xSep, self.delta_ySep, 
            fit_layer_alt=self.fit_layer_alt, fit_deltaXYseps=self.fit_deltaXYseps)
        self.total_fitting_time = time.time() - start_fit
        
        self.Cn2 = self.wind_fit.Cn2 / self.turb_results.air_mass
        self.r0 = calc_r0(self.Cn2, self.turb_results.wavelength[0])
        self.L0 = self.wind_fit.L0
        self.layer_alt = self.wind_fit.layer_alt / self.turb_results.air_mass
        self.pos_delta_xSep = self.wind_fit.pos_delta_xSep
        self.pos_delta_ySep = self.wind_fit.pos_delta_ySep
        self.wind_speed = self.wind_fit.windSpeed * self.turb_results.air_mass  #added after on-sky ss comparison results
        self.wind_direction = self.wind_fit.windDirection

        self.print_results()

        return self


        



    def print_results(self):
        print('###########################################################','\n')
        print('##################### FITTING RESULTS #####################','\n')
        print('Fitting Method: '+self.fit_method)
        print('Total Iterations : {}'.format(self.wind_fit.total_its))
        print('r0 (m) : {}'.format(self.r0))
        print('Cn2 (m^-1/3) : {}'.format(self.Cn2))
        print('L0 (m) : {}'.format(self.L0))
        print('TT Track (arcsec^2) : {}'.format(self.tt_track))
        print('LGS X Track (arcsec^2) : {}'.format(self.lgs_track.T[0]))
        print('LGS Y Track (arcsec^2) : {}'.format(self.lgs_track.T[1]))
        print('SHWFS x Shift (m) : {}'.format(self.shwfs_shift.T[0]))
        print('SHWFS y Shift (m) : {}'.format(self.shwfs_shift.T[1]))
        print('SHWFS Rotation (degrees) : {}'.format(self.shwfs_rot), '\n')
        print('Layer Altitudes: {}'.format(self.layer_alt))
        print('Delta xSep: {}'.format(self.pos_delta_xSep))
        print('Delta ySep: {}'.format(self.pos_delta_ySep))
        print('Wind Speed: {}'.format(self.wind_speed))
        print('Wind Direction: {}'.format(self.wind_direction))

        print('\n'+'###########################################################','\n')
        print('##################### TOTAL TIME TAKEN ####################','\n')
        print('############## CALCULATING OFFSETS : '+"%6.4f" % (self.time_calc_offsets)+' ###############','\n')
        print('################# FITTING OFFSETS : '+"%6.4f" % (self.total_fitting_time)+' ################', '\n')
        print('###########################################################')






    def temporal_offset_roi(self):
        
        print('\n'+'###########################################################','\n')
        print('########### PROCESSING SHWFS CENTROID OFFSETS #############','\n')
	
        if self.l3s_wind==False:
            num_arrays = 1
            cents_array = numpy.zeros((num_arrays, self.shwfs_centroids.shape[0], self.shwfs_centroids.shape[1]))
            matrix_of_offsets = numpy.zeros((num_arrays, 2, 2*numpy.sum(self.n_subap_from_pupilMask), 2*numpy.sum(self.n_subap_from_pupilMask)))
        else:
            num_arrays = 2
            cents_array = numpy.zeros((num_arrays, self.shwfs_centroids.shape[0], self.shwfs_centroids.shape[1]))
            matrix_of_offsets = numpy.zeros((num_arrays, 2, 2*numpy.sum(self.n_subap_from_pupilMask), 2*numpy.sum(self.n_subap_from_pupilMask)))
            cents_array[1] = numpy.matmul(numpy.matmul(self.l3s1_matrix, self.shwfs_centroids.T).T, self.l3s1_matrix.T)
        cents_array[0] = self.shwfs_centroids
        

        roi_width = int(2*self.wind_roi_envelope+1)
        width = int((2*self.wind_roi_envelope+1) * self.combs)
        if self.wind_map_axis=='x and y':
            length = int(self.wind_roi_belowGround + self.pupil_mask.shape[0])*2
        else:
            length = int(self.wind_roi_belowGround + self.pupil_mask.shape[0])

        if self.separate_pos_neg_offsets==True:
            roi_offsets = numpy.zeros((num_arrays, width, 2*length))
        else:
            roi_offsets = numpy.zeros((num_arrays, width, length))


        for n_a in range(num_arrays):
            #loop over how many temporal offsets are being included in the fitting procedure
            for dt in range(1, 1+self.num_offsets):
                #loop over number of GS combinations
                
                for comb in range(self.combs):

                    #induce positive and negative temporal slope offsets
                    cents_pos_offset, cents_neg_offset = self.temp_offset_cents(cents_array[n_a], 
                        dt, self.temporal_step, self.n_subap_from_pupilMask[self.selector[comb, 0]], 
                        self.n_subap_from_pupilMask[self.selector[comb, 1]], self.selector[comb])


                    #calculate roi directly from centroids or via matrix (for AOF via matrix is the fastest technique)
                    if self.roi_via_matrix==True:
                        comb_mat_pos = cross_cov(cents_pos_offset)
                        comb_map_pos = covMap_fromMatrix(comb_mat_pos, 2, self.nx_subap[:2], self.n_subap_from_pupilMask[:2], 
                            self.pupil_mask, self.wind_map_axis, self.mm, self.mmc, self.md)
                        comb_roi_pos = roi_from_map(comb_map_pos, self.gs_pos[self.selector[comb]], self.pupil_mask, self.selector[:1], 
                            self.wind_roi_belowGround, self.wind_roi_envelope)

                        comb_mat_neg = cross_cov(cents_neg_offset)
                        comb_map_neg = covMap_fromMatrix(comb_mat_neg, 2, self.nx_subap[:2], self.n_subap_from_pupilMask[:2], 
                            self.pupil_mask, self.wind_map_axis, self.mm, self.mmc, self.md)
                        comb_roi_neg = roi_from_map(comb_map_neg, self.gs_pos[self.selector[comb]], self.pupil_mask, self.selector[:1], 
                            self.wind_roi_belowGround, self.wind_roi_envelope)


                    if self.roi_via_matrix==False:
                        comb_roi_pos, time_pos = calculate_roi_covariance(cents_pos_offset, 
                            self.gs_pos[self.selector[comb]], self.pupil_mask, self.tel_diam, self.wind_roi_belowGround, 
                            self.wind_roi_envelope, self.wind_map_axis, self.wind_mapping_type)

                        comb_roi_neg, time_neg = calculate_roi_covariance(cents_neg_offset, 
                            self.gs_pos[self.selector[comb]], self.pupil_mask, self.tel_diam, self.wind_roi_belowGround, 
                            self.wind_roi_envelope, self.wind_map_axis, self.wind_mapping_type)

                    comb_roi_neg *= self.mult_neg_offset

                    if self.separate_pos_neg_offsets==True:
                        roi_offsets[n_a, comb*roi_width:(comb+1)*roi_width, :length] += comb_roi_neg
                        roi_offsets[n_a, comb*roi_width:(comb+1)*roi_width, length:] += comb_roi_pos
                    else:
                        roi_offsets[n_a, comb*roi_width:(comb+1)*roi_width] += comb_roi_neg + comb_roi_pos

            if self.include_temp0==True:
                if self.roi_via_matrix==False:
                    roi_temp0, time_temp0 = calculate_roi_covariance(cents_array[n_a], 
                        self.gs_pos, self.pupil_mask, self.tel_diam, self.wind_roi_belowGround, 
                        self.wind_roi_envelope, self.wind_map_axis, self.wind_mapping_type)
                
                if self.roi_via_matrix==True:
                    mat_temp0 = cross_cov(cents_array[n_a])
                    map_temp0 = covMap_fromMatrix(mat_temp0, self.n_wfs, self.nx_subap, self.n_subap_from_pupilMask, 
                        self.pupil_mask, self.wind_map_axis, self.mm, self.mmc, self.md)
                    roi_temp0 = roi_from_map(map_temp0, self.gs_pos, self.pupil_mask, self.selector, 
                        self.wind_roi_belowGround, self.wind_roi_envelope)


                if self.separate_pos_neg_offsets==True:
                    roi_offsets[n_a, :, :length] += roi_temp0 * self.mult_neg_offset
                    roi_offsets[n_a, :, length:] += roi_temp0
                else:
                    roi_offsets[n_a] += roi_temp0


            if self.zeroSep_cov==False:
                if self.separate_pos_neg_offsets==True:
                    roi_offsets[n_a, :, :length][self.zeroSep_locations] = 0.
                    roi_offsets[n_a, :, length:][self.zeroSep_locations] = 0.
                else:
                    roi_offsets[n_a][self.zeroSep_locations] = 0.

        return roi_offsets








    def temp_offset_cents(self, cents, dt, temporal_step, wfs1_n_subaps, wfs2_n_subaps, selector):
        """Induce a SHWFS temporal offset between SHWFS centroids
        
        Parameters:
            cents (ndarray): shwfs centroid measurements.
            dt (int): temporal step multiplier.
            temporal_step (int): temporal step
            wfs1_n_subaps (int): number of sub-apertures in wfs1.
            wfs2_n_subaps (int): number of sub-apertures in wfs2.
            selector (ndarray): wfs number of wfs1 and wfs2.

        Returns:
            ndarray: shwfs centroid measurments with negative temporal offset.
            ndarray: shwfs centroid measurments with positive temporal offset."""
        
        #Induce GS combination negative temporal shift in cents
        temp_offset_wfs1 = cents[:cents.shape[0]-dt*temporal_step, selector[0]*2*wfs1_n_subaps : selector[0]*2*wfs1_n_subaps + 2*wfs1_n_subaps]
        temp_offset_wfs2 = cents[dt*temporal_step:, selector[1]*2*wfs2_n_subaps : selector[1]*2*wfs2_n_subaps + 2*wfs2_n_subaps]
        cents_neg_offset = numpy.append(temp_offset_wfs1, temp_offset_wfs2, 1)

        #Induce GS combination positive temporal shift in cents
        temp_offset_wfs1 = cents[dt*temporal_step:, selector[0]*2*wfs1_n_subaps : selector[0]*2*wfs1_n_subaps + 2*wfs1_n_subaps]
        temp_offset_wfs2 = cents[:cents.shape[0]-dt*temporal_step, selector[1]*2*wfs2_n_subaps : selector[1]*2*wfs2_n_subaps + 2*wfs2_n_subaps]
        cents_pos_offset = numpy.append(temp_offset_wfs1, temp_offset_wfs2, 1)

        return cents_pos_offset, cents_neg_offset

