import numpy
import itertools
from scipy.misc import comb
from scipy.optimize import root, minimize
from matplotlib import pyplot; pyplot.ion()
from capt.misc_functions.calc_Cn2_r0 import calc_r0
from capt.misc_functions.mapping_matrix import get_mappingMatrix
from capt.map_functions.covMap_fromMatrix import covMap_fromMatrix
from capt.roi_functions.roi_referenceArrays import roi_referenceArrays
from capt.covariance_generation.generate_covariance_roi import covariance_roi
from capt.covariance_generation.generate_covariance_roi_l3s import covariance_roi_l3s



class fitting_parameters(object):

    def __init__(self, turb_results, fit_method, roi_offsets, frame_rate, num_offsets, offset_step, 
                wind_roi_belowGround, wind_roi_envelope, wind_map_axis, zeroSep_cov, zeroSep_locations, 
                include_temp0, mult_neg_offset, separate_pos_neg_offsets, print_fitting):

        self.turb_results = turb_results 
        self.fit_method = fit_method 
        self.roi_offsets = roi_offsets 
        self.num_offsets = num_offsets 
        self.offset_step = offset_step 
        self.wind_roi_belowGround = wind_roi_belowGround
        self.wind_roi_envelope = wind_roi_envelope
        self.wind_map_axis = wind_map_axis
        self.zeroSep_cov = zeroSep_cov
        self.zeroSep_locations = zeroSep_locations 
        self.include_temp0 = include_temp0
        self.mult_neg_offset = mult_neg_offset
        self.separate_pos_neg_offsets = separate_pos_neg_offsets
        self.print_fitting = print_fitting

        self.air_mass = self.turb_results.air_mass
        self.gs_pos = self.turb_results.gs_pos
        self.n_wfs = self.turb_results.n_wfs
        self.selector = self.turb_results.selector
        self.combs = self.turb_results.combs
        self.tel_diam = self.turb_results.tel_diam
        self.n_subap = self.turb_results.n_subap
        self.n_subap_from_pupilMask = self.turb_results.n_subap_from_pupilMask
        self.nx_subap = self.turb_results.nx_subap
        self.gs_dist = self.turb_results.gs_dist
        self.shwfs_shift = self.turb_results.shwfs_shift
        self.shwfs_rot = self.turb_results.shwfs_rot
        self.subap_diam = self.turb_results.subap_diam
        self.pupil_mask = self.turb_results.pupil_mask
        self.wavelength = self.turb_results.wavelength
        self.styc_method = self.turb_results.styc_method
        self.tt_track = self.turb_results.tt_track
        self.tt_track_present = self.turb_results.tt_track_present
        self.lgs_track = self.turb_results.lgs_track
        self.lgs_track_present = self.turb_results.lgs_track_present

        self.offset_present = self.turb_results.offset_present

        #account for air mass
        cn2 = turb_results.Cn2
        turb_cn2 = cn2.copy()
        self.Cn2 = cn2 * turb_results.air_mass
        self.Cn2[turb_cn2==self.turb_results.cn2_noiseFloor] = self.turb_results.cn2_noiseFloor  
        r0 = calc_r0(self.Cn2, self.turb_results.wavelength[0])

        self.observable_bins = self.turb_results.observable_bins
        self.delete_index = numpy.where(self.Cn2[:self.observable_bins]==self.turb_results.cn2_noiseFloor)[0]
        self.r0 = self.reduce_layers(r0, self.delete_index, self.observable_bins)
        self.L0 = self.reduce_layers(self.turb_results.L0, self.delete_index, self.observable_bins)
        self.layer_alt = self.reduce_layers(numpy.array(self.turb_results.layer_alt) * self.air_mass, 
            self.delete_index, self.observable_bins)
        
        self.n_layer = self.layer_alt.shape[0]
        self.frame_rate = frame_rate
        self.iteration = 0

        #These imported tools are the key to calculating the covariance map ROI and its analytically generated model during fitting
        onesMat, wfsMat_1, wfsMat_2, self.allMapPos, selector, self.xy_separations = roi_referenceArrays(self.pupil_mask, 
            self.gs_pos, self.tel_diam, self.wind_roi_belowGround, self.wind_roi_envelope)


        if self.wind_map_axis=='x and y':
            self.length = int(self.wind_roi_belowGround + self.pupil_mask.shape[0])*2
        else:
            self.length = int(self.wind_roi_belowGround + self.pupil_mask.shape[0])

        if self.fit_method=='L3S Fit':
            t, t, t, self.allMapPos_acrossMap, t, self.xy_separations_acrossMap = roi_referenceArrays(self.pupil_mask, 
                self.gs_pos, self.tel_diam, self.pupil_mask.shape[0]-1, self.wind_roi_envelope)





    def reduce_layers(self, param, delete_index, observable_bins):
        """Reduces fitted layer parameters to those obervable and of significant strength.
        
        Parameters:
            param (ndarray): layer parameter - r0, L0, etc.
            delete_index (ndarray): index of layers to be deleted.
            observable_bins (int): number of observable bins.
            
        Returns:
            ndarray: reduced version of param."""

        reduced_param = param[:observable_bins]
        reduced_param = numpy.delete(reduced_param, delete_index)
        return reduced_param





    def fit_roi_offsets(self, delta_xSep, delta_ySep, fit_layer_alt=False, fit_deltaXYseps=False):

        delta_xSep = self.reduce_layers(delta_xSep, self.delete_index, self.observable_bins)
        delta_ySep = self.reduce_layers(delta_ySep, self.delete_index, self.observable_bins)

        # f=pp
        if self.fit_method=='Direct Fit':
            self.direct_fit_wind(delta_xSep, delta_ySep, fit_layer_alt, fit_deltaXYseps)
        if self.fit_method=='L3S Fit':
            self.l3s_fit_wind(delta_xSep, delta_ySep, fit_layer_alt, fit_deltaXYseps)
            

        return self



    def direct_fit_wind(self, delta_xSep, delta_ySep, fit_layer_alt, fit_deltaXYseps):

        #specify which arrays are to be fitted
        try:
            len(fit_layer_alt)
        except TypeError:
            fit_layer_alt = numpy.array([fit_layer_alt]*self.n_layer)
        try:
            len(fit_deltaXYseps)
        except TypeError:
            fit_xSeps = numpy.array([fit_deltaXYseps]*self.n_layer)
            fit_ySeps = numpy.array([fit_deltaXYseps]*self.n_layer)
        
        #starting values of arrays that are to be fitted
        layer_alt = (self.layer_alt).copy().astype("object")
        delta_xSep = (delta_xSep).copy().astype("object")
        delta_ySep = (delta_ySep).copy().astype("object")

        #guess parameters to fit newAngle, layerAlt0 and layerAlt1
        guessParam = numpy.array([])
        for i, fit in enumerate(fit_layer_alt):
            if fit:
                guessParam = numpy.append(guessParam, layer_alt[i])
                layer_alt[i] = None
        for i, fit in enumerate(fit_xSeps):
            if fit:
                guessParam = numpy.append(guessParam, delta_xSep[i])
                delta_xSep[i] = None
        for i, fit in enumerate(fit_ySeps):
            if fit:
                guessParam = numpy.append(guessParam, delta_ySep[i])
                delta_ySep[i] = None


        self.generationParams = covariance_roi(self.pupil_mask, self.subap_diam, self.wavelength, 
            self.tel_diam, self.n_subap_from_pupilMask, self.gs_dist, self.gs_pos, self.n_layer, layer_alt, self.L0, 
            self.allMapPos, self.xy_separations, self.wind_map_axis, styc_method=self.styc_method, 
            wind_profiling=True, tt_track_present=self.tt_track_present, lgs_track_present=self.lgs_track_present, 
            offset_present=self.offset_present, fit_layer_alt=fit_layer_alt[0], 
            fit_tt_track=False, fit_lgs_track=False, fit_offset=False, fit_L0=False)


        static_args = (self.roi_offsets[0], self.n_layer, self.r0, self.L0, layer_alt, 
            fit_layer_alt, delta_xSep, delta_ySep, fit_deltaXYseps, 
            'Direct Fit', self.print_fitting, True)
        opPosResults = root(self.offset_fit_xySeps, guessParam, static_args, method='lm', tol=0.)
        self.total_its = self.iteration

        # pyplot.figure()
        # pyplot.imshow(self.covMapOffset)
        # pyplot.figure()
        # pyplot.imshow(self.roi_offsets[0])







    def l3s_fit_wind(self, delta_xSep, delta_ySep, fit_layer_alt, fit_deltaXYseps):
            
        #specify which arrays are to be fitted
        try:
            len(fit_layer_alt)
        except TypeError:
            fit_layer_alt = numpy.array([fit_layer_alt]*(self.n_layer-1))
        try:
            len(fit_deltaXYseps)
        except TypeError:
            fit_xSeps = numpy.array([fit_deltaXYseps]*(self.n_layer-1))
            fit_ySeps = numpy.array([fit_deltaXYseps]*(self.n_layer-1))
        
        #starting values of arrays that are to be fitted
        layer_alt_aloft = (self.layer_alt[1:]).copy().astype("object")
        delta_xSep_aloft = (delta_xSep[1:]).copy().astype("object")
        delta_ySep_aloft = (delta_ySep[1:]).copy().astype("object")

        #guess parameters to fit newAngle, layerAlt0 and layerAlt1
        guessParam_aloft = numpy.array([])
        for i, fit in enumerate(fit_layer_alt):
            if fit:
                guessParam_aloft = numpy.append(guessParam_aloft, layer_alt_aloft[i])
                layer_alt_aloft[i] = None
        for i, fit in enumerate(fit_xSeps):
            if fit:
                guessParam_aloft = numpy.append(guessParam_aloft, delta_xSep_aloft[i])
                delta_xSep_aloft[i] = None
        for i, fit in enumerate(fit_ySeps):
            if fit:
                guessParam_aloft = numpy.append(guessParam_aloft, delta_ySep_aloft[i])
                delta_ySep_aloft[i] = None

        self.generationParams = covariance_roi_l3s(self.pupil_mask, self.subap_diam, self.wavelength, 
            self.tel_diam, self.n_subap_from_pupilMask, self.gs_dist, self.gs_pos, self.n_layer-1, layer_alt_aloft, 
            self.L0[1:], self.allMapPos_acrossMap, self.xy_separations_acrossMap, self.wind_map_axis, 
            self.wind_roi_belowGround, self.wind_roi_envelope, styc_method=self.styc_method, 
            wind_profiling=True, lgs_track_present=self.lgs_track_present, offset_present=self.offset_present, 
            fit_layer_alt=fit_layer_alt[0], fit_lgs_track=False, fit_offset=False, fit_L0=False)

        static_args = (self.roi_offsets[1], self.n_layer-1, self.r0[1:], self.L0[1:], 
            layer_alt_aloft, fit_layer_alt, delta_xSep_aloft, delta_ySep_aloft, fit_deltaXYseps, 
            'L3S Fit', self.print_fitting, True)
        opPosResults = root(self.offset_fit_xySeps, guessParam_aloft, static_args, method='lm', tol=0.)
        self.total_its = self.iteration
        self.iteration = 0

        layer_alt_aloft = self.layer_alt_fit.copy()
        pos_delta_xSep_aloft = self.pos_delta_xSep.copy()
        pos_delta_ySep_aloft = self.pos_delta_ySep.copy()
        wind_speed_aloft = self.windSpeed.copy()
        wind_direction_aloft = self.windDirection.copy()
        self.covMapOffset_aloft = self.covMapOffset.copy()





        ### Generate wind ROI at altitudes of h>0km ###
        self.generationParams = covariance_roi(self.pupil_mask, self.subap_diam, self.wavelength, 
            self.tel_diam, self.n_subap_from_pupilMask, self.gs_dist, self.gs_pos, self.n_layer-1, layer_alt_aloft, 
            self.L0[1:], self.allMapPos, self.xy_separations, self.wind_map_axis, styc_method=self.styc_method, 
            wind_profiling=True, tt_track_present=False, lgs_track_present=self.lgs_track_present, 
            offset_present=self.offset_present, fit_layer_alt=fit_layer_alt[0], fit_tt_track=False, 
            fit_lgs_track=False, fit_offset=False, fit_L0=False)
        
        roi_aloft = self.offset_fit_xySeps(guessParam_aloft, self.roi_offsets[1], self.n_layer-1, 
            self.r0[1:], self.L0[1:], layer_alt_aloft, fit_layer_alt, pos_delta_xSep_aloft, 
            pos_delta_ySep_aloft, fit_deltaXYseps, 'Direct Fit', False, False)



        
        #starting values of arrays that are to be fitted
        layer_alt_ground = (self.layer_alt[:1]).copy().astype("object")
        delta_xSep_ground = (delta_xSep[:1]).copy().astype("object")
        delta_ySep_ground = (delta_ySep[:1]).copy().astype("object")

        #guess parameters to fit newAngle, layerAlt0 and layerAlt1
        guessParam_ground = numpy.array([])
        for i, fit in enumerate(fit_layer_alt[:1]):
            if fit:
                guessParam_ground = numpy.append(guessParam_ground, layer_alt_ground[i])
                layer_alt_ground[i] = None
        for i, fit in enumerate(fit_xSeps[:1]):
            if fit:
                guessParam_ground = numpy.append(guessParam_ground, delta_xSep_ground[i])
                delta_xSep_ground[i] = None
        for i, fit in enumerate(fit_ySeps[:1]):
            if fit:
                guessParam_ground = numpy.append(guessParam_ground, delta_ySep_ground[i])
                delta_ySep_ground[i] = None


        # pyplot.figure()
        # pyplot.imshow(self.roi_offsets[0])
        # pyplot.figure()
        # pyplot.imshow(roi_aloft)
        # pyplot.figure()
        # pyplot.imshow(self.roi_offsets[0]-roi_aloft)

        self.generationParams = covariance_roi(self.pupil_mask, self.subap_diam, self.wavelength, 
            self.tel_diam, self.n_subap_from_pupilMask, self.gs_dist, self.gs_pos, 1, layer_alt_ground, self.L0[:1], 
            self.allMapPos, self.xy_separations, self.wind_map_axis, styc_method=self.styc_method, 
            wind_profiling=True, tt_track_present=self.tt_track_present, lgs_track_present=self.lgs_track_present, 
            offset_present=self.offset_present, fit_layer_alt=fit_layer_alt[0], fit_tt_track=False, 
            fit_lgs_track=False, fit_offset=False, fit_L0=False)

        static_args = (self.roi_offsets[0]-roi_aloft, 1, self.r0[:1], self.L0[:1], layer_alt_ground, 
            fit_layer_alt, delta_xSep_ground, delta_ySep_ground, fit_deltaXYseps, 
            'Direct Fit', self.print_fitting, True)
        opPosResults = root(self.offset_fit_xySeps, guessParam_ground, static_args, method='lm', tol=0.)
        self.total_its += self.iteration

        self.layer_alt = numpy.append(self.layer_alt_fit, layer_alt_aloft)
        self.pos_delta_xSep = numpy.append(self.pos_delta_xSep, pos_delta_xSep_aloft)
        self.pos_delta_ySep = numpy.append(self.pos_delta_ySep, pos_delta_ySep_aloft)
        self.windSpeed = numpy.append(self.windSpeed, wind_speed_aloft)
        self.windDirection = numpy.append(self.windDirection, wind_direction_aloft)
        self.covMapOffset += roi_aloft






    def offset_fit_xySeps(self, guessParam, target, n_layer, r0, L0, layer_alt, fit_layer_alt, delta_xSep, 
            delta_ySep, fit_deltaXYseps, fit_method, print_fitting, output_residual):
        
        #assign parameters to be fitted with respective guessPos parameters
        np=0

        layer_alt = layer_alt.copy()
        #assign fitLayerAlt guess values
        for i , val in enumerate(layer_alt):																
            if val==None:
                layer_alt[i] = numpy.abs(guessParam[np])
                np+=1

        delta_xSep = delta_xSep.copy()
        #assign fitLayerAlt guess values
        for i , val in enumerate(delta_xSep):		
            if val==None:
                delta_xSep[i] = guessParam[np]
                np+=1

        delta_ySep = delta_ySep.copy()
        #assign fitLayerAlt guess values
        for i , val in enumerate(delta_ySep):					
            if val==None:
                delta_ySep[i] = guessParam[np]
                np+=1


        #reset self.covMapOffset after each iteration
        self.covMapOffset = numpy.zeros(self.roi_offsets[0].shape)
        self.layer_alt_fit = layer_alt.astype('float')
        self.pos_delta_xSep = delta_xSep.astype('float')
        self.pos_delta_ySep = delta_ySep.astype('float')
        self.windSpeed = numpy.sqrt(self.pos_delta_xSep**2 + self.pos_delta_ySep**2) * (self.frame_rate/self.offset_step)	
        self.windDirection = 360 - self.xySep_vectorAngle(self.pos_delta_ySep, self.pos_delta_xSep) * 180/numpy.pi


        #generate turbulence profile at dt=0 with fitLayerAlt0 altitudes - if includeTemp0=True
        if self.include_temp0==True:
            if fit_layer_alt[0]==True or self.iteration==0:
                
                if fit_method=='Direct Fit':
                    self.roi_temp0 = self.generationParams._make_covariance_roi_(self.layer_alt_fit, 
                        r0, L0, tt_track=self.tt_track, lgs_track=self.lgs_track, shwfs_shift=self.shwfs_shift, shwfs_rot=self.shwfs_rot, 
                        delta_xSep=self.pos_delta_xSep*0, delta_ySep=self.pos_delta_ySep*0)
                
                if fit_method=='L3S Fit':
                    self.roi_temp0 = self.generationParams._make_covariance_roi_l3s_(self.layer_alt_fit, 
                        r0, L0, lgs_track=self.lgs_track, shwfs_shift=self.shwfs_shift, shwfs_rot=self.shwfs_rot, 
                        delta_xSep=self.pos_delta_xSep*0, delta_ySep=self.pos_delta_ySep*0)

            if self.separate_pos_neg_offsets==True:
                self.covMapOffset[:, :self.length] += self.roi_temp0 * self.mult_neg_offset
                self.covMapOffset[:, self.length:] += self.roi_temp0
            else:
                self.covMapOffset += self.roi_temp0

        

        if self.iteration==0 or fit_deltaXYseps==True:

            for n in range(self.num_offsets):
                inter_delta_xSep_posi = (n+1) * self.pos_delta_xSep / self.num_offsets
                inter_delta_ySep_posi = (n+1) * self.pos_delta_ySep / self.num_offsets

                if fit_method=='Direct Fit':
                    pos_roi_wind_fit = self.generationParams._make_covariance_roi_(self.layer_alt_fit, 
                        r0, L0, tt_track=self.tt_track, lgs_track=self.lgs_track, shwfs_shift=self.shwfs_shift, shwfs_rot=self.shwfs_rot, 
                        delta_xSep=inter_delta_xSep_posi, delta_ySep=inter_delta_ySep_posi)
                    neg_roi_wind_fit = self.generationParams._make_covariance_roi_(self.layer_alt_fit, 
                        r0, L0, tt_track=self.tt_track, lgs_track=self.lgs_track, shwfs_shift=self.shwfs_shift, shwfs_rot=self.shwfs_rot, 
                        delta_xSep=-inter_delta_xSep_posi, delta_ySep=-inter_delta_ySep_posi)

                if fit_method=='L3S Fit':
                    pos_roi_wind_fit = self.generationParams._make_covariance_roi_l3s_(self.layer_alt_fit, 
                        r0, L0, lgs_track=self.lgs_track, shwfs_shift=self.shwfs_shift, shwfs_rot=self.shwfs_rot, 
                        delta_xSep=inter_delta_xSep_posi, delta_ySep=inter_delta_ySep_posi)
                    neg_roi_wind_fit = self.generationParams._make_covariance_roi_l3s_(self.layer_alt_fit, 
                        r0, L0, lgs_track=self.lgs_track, shwfs_shift=self.shwfs_shift, shwfs_rot=self.shwfs_rot, 
                        delta_xSep=-inter_delta_xSep_posi, delta_ySep=-inter_delta_ySep_posi)

                if self.separate_pos_neg_offsets==True:
                    self.covMapOffset[:, :self.length] += neg_roi_wind_fit * self.mult_neg_offset
                    self.covMapOffset[:, self.length:] += pos_roi_wind_fit
                else:
                    self.covMapOffset += pos_roi_wind_fit + (neg_roi_wind_fit*self.mult_neg_offset)

        if self.zeroSep_cov==False:
            if self.separate_pos_neg_offsets==True:
                self.covMapOffset[:, :self.length][self.zeroSep_locations] = 0.
                self.covMapOffset[:, self.length:][self.zeroSep_locations] = 0.
            else:
                self.covMapOffset[self.zeroSep_locations] = 0.

        if print_fitting==True:
            print('\n')
            print("Iteration: {}".format(self.iteration))
            print('Method: {}'.format(fit_method))
            print("Layer Distance: {}".format(self.layer_alt_fit))
            print("L0: {}".format(L0))
            print("r0: {}".format(r0))
            print('Num. Offsets: {}'.format(self.num_offsets))
            print('Offset Step: {}'.format(self.offset_step))
            print("Delta xSep: {}".format(delta_xSep))
            print("Delta ySep: {}".format(delta_ySep))
            print("Wind Speed: {}".format(self.windSpeed))
            print("Wind Direction: {}".format(self.windDirection))


        # cov_map_image = self.covMapOffset.copy()

        # fc, axesc = pyplot.subplots()
        # cov_map_image[numpy.where(self.dummy_map==0)]=numpy.nan

        # imc = axesc.imshow(numpy.rot90(numpy.fliplr(cov_map_image), 1), vmin=-0.012783397622145072, vmax=0.00884477409894946)
        # xticklabels = (numpy.linspace(-6,6,7)).astype('int')
        # xticks = numpy.linspace(0,12,7)
        # axesc.set_xticks(xticks)
        # axesc.set_xticklabels(xticklabels)
        # yticklabels = numpy.linspace(-6,6,7).astype('int')
        # yticks = numpy.linspace(0,12,7)
        # axesc.set_yticks(yticks)
        # axesc.set_yticklabels(yticklabels)
        # pyplot.ylabel('Sub-aperture Separation, $y$')
        # pyplot.xlabel('Sub-aperture Separation, $x$')
        # pyplot.title('$-(-\delta t) + (+\delta t)$')
        # pyplot.plot([12.5]*12, numpy.linspace(-0.5,12.5,12), color='k')

        # pyplot.plot(numpy.array([-0.5,12.5]), numpy.array([6,6.]), linestyle='--', color='w')
        # pyplot.plot(numpy.array([6,6]), numpy.array([-0.5,6.]), linestyle='--', color='w')
        # pyplot.plot(numpy.array([6,6]), numpy.array([6,12.5]), linestyle='--', color='limegreen')


        # pyplot.savefig('pngs_forVideos/wind_fit/wind'+str(self.iteration)+'.png', dpi=800)



        if output_residual==True:
            self.iteration += 1
            return numpy.sqrt((self.covMapOffset - target)**2).flatten()
        else:
            return self.covMapOffset




    def sortArraysByAltitude(self):
        if self.windDirect==True:
            #gets index of lowest->highest of fitted altitudes at dt=0 
            sorted_index = numpy.argsort(self.direct_layerAlt0)

            self.direct_layerAlt0 = self.direct_layerAlt0[sorted_index]
            self.direct_delta_xSep_posi = self.direct_delta_xSep_posi[sorted_index]
            self.direct_delta_ySep_posi = self.direct_delta_ySep_posi[sorted_index]
            self.direct_r0 = self.r0[sorted_index]
            self.direct_L0 = self.L0[sorted_index]
            self.direct_windSpeed = self.direct_windSpeed[sorted_index]
            self.direct_windDirection = self.direct_windDirection[sorted_index]


        if self.windL3S==True:
            #gets index of lowest->highest of fitted altitudes at dt=0 
            sorted_index = numpy.argsort(self.l3s_layerAlt0)

            self.l3s_layerAlt0 = self.l3s_layerAlt0[sorted_index]
            self.l3s_delta_xSep_posi = self.l3s_delta_xSep_posi[sorted_index]
            self.l3s_delta_ySep_posi = self.l3s_delta_ySep_posi[sorted_index]
            self.l3s_r0 = self.r0[sorted_index]
            self.l3s_L0 = self.L0[sorted_index]
            self.l3s_windSpeed = self.l3s_windSpeed[sorted_index]
            self.l3s_windDirection = self.l3s_windDirection[sorted_index]




    def xySep_vectorAngle(self, dx, dy):
        """Calculate direction of dx/dy displacement.
        
        Parameters:
            dx (ndarray): displacement in x.
            dy (ndarray): displacement in y.
            
        Returns:
            ndarray: direction of displacement"""
        
        dx[dx==0] = 1e-20
        theta = (numpy.pi/2.) - numpy.arctan(dy/dx)

        check = numpy.zeros(dx.shape)
        check[dx<0] += 1
        check[dy>=0] += 1

        theta[check==2] = numpy.pi + ((numpy.pi/2.) - numpy.arctan(dy[check==2]/dx[check==2]))

        check *= 0
        check[dx<0] += 1
        check[dy<0] += 1
        theta[check==2] = 3*(numpy.pi)/2. - numpy.arctan(dy[check==2]/dx[check==2])
        
        return theta