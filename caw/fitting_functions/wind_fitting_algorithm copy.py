import numpy
import math
import itertools
from scipy.misc import comb
# from gsmaxAlt import gsmaxAlt
from covSliceZeros import sliceZeros
from observableAlts import observableAlts
from gsSeparationAngle import gsSeparationAngle
from scipy.optimize import root, minimize
from covMapsFromMatrix import covMapsFromMatrix
from matplotlib import pyplot; pyplot.ion()
from covMapVector import vectorAngle
from covarianceSlice_referenceArrays import referenceArrays
from covSliceTrackMatrix import sliceTrackMatrix
from generateCovarianceMatrix import CovarianceMatrix
from covMapsFromMatrix import covMapsFromMatrix
from covSlicesFromMaps import covSlicesFromMaps
from superFastCovMap import getMappingMatrix, superFastCovMap
from transformMatrix import transformMatrix

from generateCovarianceSlice_wind_xySeps import CovarianceSlice as CovarianceSlice_test
from generateCovarianceSliceL3sStep1_wind_xySeps import covarianceSliceL3sStep1 as covarianceSliceL3sStep1_test


class windProfilingParams(object):

    def __init__(self, i, turbProfilingParams, windDirect, windL3S, mapOffset, mapAloftOffset, mapOnes, layerAlt, airMass, r0, L0, track, pupilShift, pupilRot, gsPos, noOffsets, offsetStep, 
                windBelowGround, windEnvelope, windSliceAxis, windZeroSeps, includeTemp0, multNegativeTempOffset):

        self.dataNum = i
        self.turbProfilingParams = turbProfilingParams
        self.windDirect = windDirect
        self.windL3S = windL3S
        self.windSliceAxis = windSliceAxis
        self.mapOffset = mapOffset
        self.mapAloftOffset = mapAloftOffset

        self.mapOnes = mapOnes
        #layers to be used at dt=0
        self.layerAlt0 = layerAlt.copy()
        #layers to be used at |dt|>0
        self.layerAlt1 = layerAlt.copy()
        self.nLayers = len(layerAlt)
        self.nLayersAloft = len(layerAlt) - 1
        self.airMass = airMass
        self.r0 = r0
        self.L0 = L0
        self.noOffsets = noOffsets
        self.offsetStep = offsetStep
        self.gsPos = gsPos
        self.n_wfs = turbProfilingParams.nWfs
        self.selector = turbProfilingParams.selector
        self.combs = turbProfilingParams.combs
        self.telDiam = turbProfilingParams.telDiam
        self.nSubaps = turbProfilingParams.nSubaps
        self.nxSubaps = turbProfilingParams.nxSubaps
        self.gsAlt = turbProfilingParams.gsAlt
        self.pupilShift = pupilShift
        self.pupilRot = pupilRot
        self.subapDiam = turbProfilingParams.subapDiam
        self.pupilMask = turbProfilingParams.pupilMask
        self.wfsWavelength = turbProfilingParams.wfsWavelength
        self.fitOffsets = turbProfilingParams.fitOffsets
        self.stycMethod = turbProfilingParams.stycMethod
        self.includeTemp0 = includeTemp0
        self.multNegativeTempOffset = multNegativeTempOffset
        ones = numpy.ones((self.nSubaps, self.nSubaps))
        self.mm, self.mmc, self.md = getMappingMatrix(self.pupilMask, ones)
        self.transformMatrix_2GS = transformMatrix(self.nSubaps, 2)

        self.wfsFreq = 1./turbProfilingParams.loopTime
        self.windBelowGround = windBelowGround
        self.windEnvelope = windEnvelope
        self.windZeroSeps = windZeroSeps
        self.iterationWind = 0

        #These imported tools are the key to calculating the covariance map ROI and its analytically generated model during fitting
        onesMat, wfsMat_1, wfsMat_2, self.allMapPos, selector, self.xy_separations = referenceArrays(self.pupilMask, gsPos, self.telDiam, self.windBelowGround, self.windEnvelope)
        self.mapWidth = self.xy_separations.shape[1]
        # self.singleMapWidth = self.xy_separations.shape[1]
        # minAlt, maxAlt = observableAlts(self.gsPos, self.telDiam)
        self.maxAlt = numpy.zeros(self.combs)
        self.gamma0 = numpy.zeros(self.combs)
        self.vectorLength = numpy.zeros(self.combs)
        
        # oneTrack = numpy.array([1.])
        self.sliceWidth = self.xy_separations.shape[1]
        self.sliceLength = self.xy_separations.shape[2]
        self.trackValues = track
        self.track = sliceTrackMatrix(self.combs, self.sliceWidth, self.sliceLength, self.trackValues)
        self.track = numpy.concatenate((self.track, self.track), axis=1) 

        for i in range(self.combs):
            relGsPos = numpy.vstack((self.gsPos[self.selector[i][0]], self.gsPos[self.selector[i][1]]))

            self.gamma0[i] = vectorAngle(relGsPos[0], relGsPos[1])
            minAlt, self.maxAlt[i] = observableAlts(relGsPos, self.telDiam)
            self.vectorLength[i] = numpy.sqrt(numpy.sum((relGsPos[0]-relGsPos[1])**2.)) 

        self.multSlice = 1.
        if self.windZeroSeps == False:
            self.multSlice = sliceZeros(self.combs, self.xy_separations.shape[1], self.xy_separations.shape[2], self.windSliceAxis, self.windBelowGround)
            self.multSlice = numpy.concatenate((self.multSlice, self.multSlice), axis=1)
            self.mapOffset *= self.multSlice
            self.mapAloftOffset *= self.multSlice
            self.mapOnes *= self.multSlice
            self.track *= self.multSlice

        if self.windL3S==True:
            t, t, t, t, t, self.xy_separations_acrossMap = referenceArrays(self.pupilMask, gsPos, self.telDiam, self.pupilMask.shape[0]-1, self.windEnvelope)




    def fitMapOffsets(self, fit_layerAlt0=False, fit_deltaXYseps=False):

        self.fittingArray = numpy.array([fit_layerAlt0, fit_deltaXYseps])

        if self.windDirect==True:
            #specify which arrays are to be fitted
            try:
                len(fit_layerAlt0)
            except TypeError:
                fit_layerAlt0 = numpy.array([fit_layerAlt0]*self.nLayers)
            try:
                len(fit_deltaXYseps)
            except TypeError:
                fit_xSeps = numpy.array([fit_deltaXYseps]*self.nLayers)
                fit_ySeps = numpy.array([fit_deltaXYseps]*self.nLayers)
            
            #starting values of arrays that are to be fitted
            layerAlt0 = (self.layerAlt0).copy().astype("object")
            delta_xSep = (numpy.array([0.]*self.nLayers)).copy().astype("object")
            delta_ySep = (numpy.array([0.]*self.nLayers)).copy().astype("object")

            #guess parameters to fit newAngle, layerAlt0 and layerAlt1
            guessParam = numpy.array([])
            for i, fit in enumerate(fit_layerAlt0):
                if fit:
                    guessParam = numpy.append(guessParam, layerAlt0[i])
                    layerAlt0[i] = None
            for i, fit in enumerate(fit_xSeps):
                if fit:
                    guessParam = numpy.append(guessParam, delta_xSep[i])
                    delta_xSep[i] = None
            for i, fit in enumerate(fit_ySeps):
                if fit:
                    guessParam = numpy.append(guessParam, delta_ySep[i])
                    delta_ySep[i] = None
        
            opPosResults = root(self.temporalOffsetMapFitting_xySeps, guessParam, args = (self.mapOffset, self.nLayers, 
                        self.r0, self.L0, layerAlt0, delta_xSep, delta_ySep, 'windDirect', True), method='lm', tol=0.)
            self.iterationWind = 0

            self.direct_layerAlt0 = self.layerAlt0.copy()
            self.direct_delta_xSep_posi = self.delta_xSep_posi.copy()
            self.direct_delta_ySep_posi = self.delta_ySep_posi.copy()
            self.direct_windSpeed = self.windSpeed
            self.direct_windDirection = self.windDirection
            self.direct_fittedMapOffset = self.covMapOffset.copy()

            # # calculates wind speed/direction from each layer's spatial offset within the map, under some temporal offset in slopes 
            # self.direct_windSpeed, self.direct_windDirection = self.windProfileInformation(self.direct_delta_xSep_posi, self.direct_delta_ySep_posi)

        


        if self.windL3S==True:

            #specify which aloft arrays are to be fitted
            try:
                len(self.fittingArray[0])
            except TypeError:
                fit_aloft_layerAlt0 = numpy.array([self.fittingArray[0]]*(self.nLayers-1))
            try:
                len(self.fittingArray[1])
            except TypeError:
                fit_aloft_xSeps = numpy.array([self.fittingArray[1]]*(self.nLayers-1))
                fit_aloft_ySeps = numpy.array([self.fittingArray[1]]*(self.nLayers-1))
            
            #starting values of arrays that are to be fitted
            aloft_layerAlt0 = (self.layerAlt0[1:]).copy().astype("object")
            aloft_delta_xSep = (numpy.array([0.]*(self.nLayers-1))).copy().astype("object")
            aloft_delta_ySep = (numpy.array([0.]*(self.nLayers-1))).copy().astype("object")

            #guess parameters to fit newAngle, layerAlt0 and layerAlt1
            aloft_guessParam = numpy.array([])
            for i, fit in enumerate(fit_aloft_layerAlt0):
                if fit:
                    aloft_guessParam = numpy.append(aloft_guessParam, aloft_layerAlt0[i])
                    aloft_layerAlt0[i] = None
            for i, fit in enumerate(fit_aloft_xSeps):
                if fit:
                    aloft_guessParam = numpy.append(aloft_guessParam, aloft_delta_xSep[i])
                    aloft_delta_xSep[i] = None
            for i, fit in enumerate(fit_aloft_ySeps):
                if fit:
                    aloft_guessParam = numpy.append(aloft_guessParam, aloft_delta_ySep[i])
                    aloft_delta_ySep[i] = None

            #specify which ground arrays are to be fitted
            try:
                len(self.fittingArray[0])
            except TypeError:
                fit_ground_layerAlt0 = numpy.array([self.fittingArray[0]])
            try:
                len(self.fittingArray[1])
            except TypeError:
                fit_ground_xSeps = numpy.array([self.fittingArray[1]])
                fit_ground_ySeps = numpy.array([self.fittingArray[1]])
            
            #starting values of arrays that are to be fitted
            ground_layerAlt0 = (self.layerAlt0[:1]).copy().astype("object")
            ground_delta_xSep = (numpy.array([0.])).copy().astype("object")
            ground_delta_ySep = (numpy.array([0.])).copy().astype("object")

            #guess parameters to fit newAngle, layerAlt0 and layerAlt1
            ground_guessParam = numpy.array([])
            for i, fit in enumerate(fit_ground_layerAlt0):
                if fit:
                    ground_guessParam = numpy.append(ground_guessParam, ground_layerAlt0[i])
                    ground_layerAlt0[i] = None
            for i, fit in enumerate(fit_ground_xSeps):
                if fit:
                    ground_guessParam = numpy.append(ground_guessParam, ground_delta_xSep[i])
                    ground_delta_xSep[i] = None
            for i, fit in enumerate(fit_ground_ySeps):
                if fit:
                    ground_guessParam = numpy.append(ground_guessParam, ground_delta_ySep[i])
                    ground_delta_ySep[i] = None

            opPosResults = root(self.temporalOffsetMapFitting_xySeps, aloft_guessParam, args = (self.mapAloftOffset, self.nLayers-1, 
                        self.r0[1:], self.L0[1:], aloft_layerAlt0, aloft_delta_xSep, aloft_delta_ySep, 'windL3S', True), method='lm', tol=0.)
            self.iterationWind = 0

            self.aloft_layerAlt0 = self.layerAlt0.copy()
            self.aloft_delta_xSep_posi = self.delta_xSep_posi.copy()
            self.aloft_delta_ySep_posi = self.delta_ySep_posi.copy()
            self.aloft_windSpeed = self.windSpeed.copy()
            self.aloft_windDirection = self.windDirection.copy()
            self.aloft_fittedMapOffset = self.covMapOffset.copy()

            # pyplot.figure()
            # pyplot.imshow(self.aloft_fittedMapOffset)
            # d=n
            
            mapOffset_aloft = self.generateFittedTemporalOffset(self.nLayers-1, self.r0[1:], self.L0[1:], self.layerAlt0, self.delta_xSep_posi, 
                        self.delta_ySep_posi, self.fittingArray, 'windDirect')


            opPosResults = root(self.temporalOffsetMapFitting_xySeps, ground_guessParam, args = (self.mapOffset - mapOffset_aloft, 1, 
                        self.r0[:1], self.L0[:1], ground_layerAlt0, ground_delta_xSep, ground_delta_ySep, 'windDirect', True), method='lm', tol=0.)
            self.iterationWind = 0

            self.l3s_layerAlt0 = numpy.append(self.layerAlt0, self.aloft_layerAlt0)
            self.l3s_delta_xSep_posi = numpy.append(self.delta_xSep_posi, self.aloft_delta_xSep_posi)
            self.l3s_delta_ySep_posi = numpy.append(self.delta_ySep_posi, self.aloft_delta_ySep_posi)
            self.l3s_windSpeed = numpy.append(self.windSpeed, self.aloft_windSpeed)
            self.l3s_windDirection = numpy.append(self.windDirection, self.aloft_windDirection)



        #if fitLayerAlt0=True, the ordering of the input layers may change. This function sorts parameters by ascending altitude.
        self.sortArraysByAltitude()


        if self.windDirect==True:
            print('\n')
            print('DIRECT FITTING')
            print("r0: {}".format(self.direct_r0))
            print("Wind Velocity: {}".format(self.direct_windSpeed))
            print("Wind Direction: {}".format(self.direct_windDirection))
            layerAlt0 = self.direct_layerAlt0
            r0 = self.direct_r0
            L0 = self.direct_L0
            windSpeed = self.direct_windSpeed
            windDirection = self.direct_windDirection

            pyplot.figure('Direct Covariance')
            pyplot.imshow(self.mapOffset, interpolation='nearest')
            pyplot.figure('Overall Fit')
            pyplot.imshow(self.direct_fittedMapOffset, interpolation='nearest')

        if self.windL3S==True:
            print('\n')
            print('\n')
            print('L3S FITTING')
            print("Altitude (km): {}".format(self.l3s_layerAlt0/1000.))
            print('\n')
            print("Track (acrsecs): {}".format(self.trackValues))
            print("L0 Profile (m): {}".format(self.L0))
            print("Turb Profile (r0): {}".format(self.l3s_r0))
            print('\n')
            print("Wind Velocity (m/s): {}".format(self.l3s_windSpeed))
            print("Wind Direction (degrees): {}".format(self.l3s_windDirection))
            layerAlt0 = self.l3s_layerAlt0
            r0 = self.l3s_r0
            L0 = self.l3s_L0
            windSpeed = self.l3s_windSpeed
            windDirection = self.l3s_windDirection

            pyplot.figure('L3S.1 Covariance')
            pyplot.imshow(self.mapAloftOffset, interpolation='nearest')
            pyplot.figure('L3S.1 Overall Fit')
            pyplot.imshow(self.aloft_fittedMapOffset)            

        return layerAlt0, r0, L0, windSpeed, windDirection


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



    def temporalOffsetMapFitting_xySeps(self, guessParam, target, n_layers, r0, L0, layerAlt0, delta_xSep, 
            delta_ySep, fittingMethod, outputResidual):
        
        #assign parameters to be fitted with respective guessPos parameters
        np=0

        layerAlt0 = layerAlt0.copy()
        for i , val in enumerate(layerAlt0):																#assign fitLayerAlt guess values
            if val==None:
                layerAlt0[i] = guessParam[np]
                np+=1

        delta_xSep = delta_xSep.copy()
        for i , val in enumerate(delta_xSep):																#assign fitLayerAlt guess values
            if val==None:
                delta_xSep[i] = guessParam[np]
                np+=1

        delta_ySep = delta_ySep.copy()
        for i , val in enumerate(delta_ySep):																#assign fitLayerAlt guess values
            if val==None:
                delta_ySep[i] = guessParam[np]
                np+=1

        #reset self.covMapOffset after each iteration
        self.covMapOffset = numpy.zeros(self.mapOffset.shape)
        self.delta_xSep_posi = delta_xSep.astype('float')
        self.delta_ySep_posi = delta_ySep.astype('float')

        print('\n')
        print('Fitting Method: {}'.format(fittingMethod))
        print('Data No.: {}'.format(self.dataNum))
        print('No. Offsets: {}'.format(self.noOffsets))
        print('Offset Step: {}'.format(self.offsetStep))
        print("r0: {}".format(r0))
        print("L0: {}".format(L0))
        print("Altitude0: {}".format(layerAlt0))
        print("Delta xSep: {}".format(delta_xSep))
        print("Delta ySep: {}".format(delta_ySep))
        print("Iteration: {}".format(self.iterationWind))

        #generate turbulence profile at dt=0 with fitLayerAlt0 altitudes - if includeTemp0=True
        if self.includeTemp0 == True:
            if self.fittingArray[0]==True or self.iterationWind==0:
                if fittingMethod=='windDirect':
                    generationParams_0 = CovarianceSlice_test(self.pupilMask, self.subapDiam, self.wfsWavelength, self.telDiam, self.gsAlt, self.gsPos, self.allMapPos, 
                        self.xy_separations, n_layers, layerAlt0*self.airMass, False, self.fitOffsets, False, L0, self.windSliceAxis, self.stycMethod)
                if fittingMethod=='windL3S':
                    generationParams_0 = covarianceSliceL3sStep1_test(self.pupilMask, self.subapDiam, self.wfsWavelength, self.telDiam, self.gsAlt, self.gsPos, 
                        self.xy_separations_acrossMap, n_layers, layerAlt0, False, self.windBelowGround, self.windEnvelope, False, L0, self.windSliceAxis, self.stycMethod)
                self.mapOffset_temp0 = generationParams_0._make_covariance_slice(r0, L0, self.pupilShift, self.pupilRot)
            self.covMapOffset[:, :self.sliceLength] += self.mapOffset_temp0
            self.covMapOffset[:, self.sliceLength:] += self.mapOffset_temp0

        displacement = numpy.sqrt(self.delta_xSep_posi**2 + self.delta_ySep_posi**2)
        self.windSpeed = displacement*(self.wfsFreq/self.offsetStep)	
        print("Wind Speed: {}".format(self.windSpeed))
        
        self.windDirection = 360 - self.xySep_vectorAngle(self.delta_ySep_posi, self.delta_xSep_posi) * 180/numpy.pi
        # theta = numpy.arcsin(self.delta_ySep_posi / numpy.sqrt(self.delta_ySep_posi**2. + self.delta_xSep_posi**2.))
        # wd = ((self.gamma0[0] - theta - numpy.pi) * 180/numpy.pi) + 360
        # print(theta* 180/numpy.pi, self.gamma0* 180/numpy.pi)
        # self.windDirection = (numpy.modf(wd/360.)[0])*360.
        # self.windDirection = (self.gamma0[0] - vectorAngle(numpy.array([self.delta_ySep_posi,self.delta_xSep_posi]), numpy.array([0.,0.]))) * 180./numpy.pi
        print("Wind Direction: {}".format(self.windDirection))


        if self.iterationWind==0 or self.fittingArray[1]==True:
            if fittingMethod=='windDirect':
                generationParams_aloft = CovarianceSlice_test(self.pupilMask, self.subapDiam, self.wfsWavelength, self.telDiam, self.gsAlt, self.gsPos, 
                            self.allMapPos, self.xy_separations, n_layers, layerAlt0*self.airMass, True, self.fitOffsets, False, L0, self.windSliceAxis, self.stycMethod)
            if fittingMethod=='windL3S':
                 generationParams_aloft = covarianceSliceL3sStep1_test(self.pupilMask, self.subapDiam, self.wfsWavelength, self.telDiam, self.gsAlt, self.gsPos, 
                            self.xy_separations_acrossMap, n_layers, layerAlt0, True, self.windBelowGround, self.windEnvelope, False, L0, self.windSliceAxis, self.stycMethod)

        self.covMapOffset[:, :self.sliceLength*2] += generationParams_aloft._make_covariance_slice(r0, L0, self.pupilShift, self.pupilRot, delta_xSep=-self.delta_xSep_posi, 
                        delta_ySep=-self.delta_ySep_posi) * self.multNegativeTempOffset
        self.covMapOffset[:, self.sliceLength*2:] += generationParams_aloft._make_covariance_slice(r0, L0, self.pupilShift, self.pupilRot, delta_xSep=self.delta_xSep_posi, 
                        delta_ySep=self.delta_ySep_posi)

        if self.noOffsets>1:
            for n in range(self.noOffsets-1):
                inter_delta_xSep_posi = (n+1) * self.delta_xSep_posi / self.noOffsets
                inter_delta_ySep_posi = (n+1) * self.delta_ySep_posi / self.noOffsets
                
                self.covMapOffset[:, :self.sliceLength*2] += generationParams_aloft._make_covariance_slice(r0, L0, self.pupilShift, self.pupilRot, delta_xSep=-inter_delta_xSep_posi, 
                                delta_ySep=-inter_delta_ySep_posi) * self.multNegativeTempOffset
                self.covMapOffset[:, self.sliceLength*2:] += generationParams_aloft._make_covariance_slice(r0, L0, self.pupilShift, self.pupilRot, delta_xSep=inter_delta_xSep_posi, 
                                delta_ySep=inter_delta_ySep_posi)

        if fittingMethod=='windDirect':
            self.covMapOffset += self.track
        self.covMapOffset *= self.multSlice

        if outputResidual==True:
            self.iterationWind += 1
            self.layerAlt0 = layerAlt0.astype('float')
            # return numpy.sqrt((self.covMapOffset - target)**2).flatten()
            return numpy.sqrt((self.covMapOffset[self.mapOnes==1] - target[self.mapOnes==1])**2).flatten()
            # return (numpy.sqrt(self.covMapOffset**2) - numpy.sqrt(target**2)).flatten()
        else:
            return self.covMapOffset




    def generateFittedTemporalOffset(self, n_layers, r0, L0, layerAlt0, delta_xSep, delta_ySep, fittingArray, generationMethod):
        
        
        #specify which arrays are to be fitted
        try:
            len(fittingArray[0])
        except TypeError:
            fit_layerAlt0 = numpy.array([fittingArray[0]]*n_layers)
        try:
            len(fittingArray[1])
        except TypeError:
            fit_xSeps = numpy.array([fittingArray[1]]*n_layers)
            fit_ySeps = numpy.array([fittingArray[1]]*n_layers)
        
        #starting values of arrays that are to be fitted
        layerAlt0 = layerAlt0.copy().astype("object")
        delta_xSep = delta_xSep.copy().astype("object")
        delta_ySep = delta_ySep.copy().astype("object")

        #guess parameters to fit newAngle, layerAlt0 and layerAlt1
        fittedParam = numpy.array([])
        for i, fit in enumerate(fit_layerAlt0):
            if fit:
                fittedParam = numpy.append(fittedParam, layerAlt0[i])
                layerAlt0[i] = None
        for i, fit in enumerate(fit_xSeps):
            if fit:
                fittedParam = numpy.append(fittedParam, delta_xSep[i])
                delta_xSep[i] = None
        for i, fit in enumerate(fit_ySeps):
            if fit:
                fittedParam = numpy.append(fittedParam, delta_ySep[i])
                delta_ySep[i] = None

        fittedSlice_temporalOffsets = self.temporalOffsetMapFitting_xySeps(fittedParam, False, n_layers, r0, L0, layerAlt0, delta_xSep, delta_ySep, generationMethod, False)
        return fittedSlice_temporalOffsets




    def xySep_vectorAngle(self, dx, dy):

        theta = (numpy.pi/2.) - numpy.arctan(dy/dx)

        check = numpy.zeros(dx.shape)
        check[dx<0] += 1
        check[dy>=0] += 1

        theta[check==2] = numpy.pi + ((numpy.pi/2.) - numpy.arctan(dy[check==2]/dx[check==2]))

        # if dx<0 and dy>=0:
        #     theta = numpy.pi + ((numpy.pi/2.) - numpy.arctan(dy/dx))

        check *= 0
        check[dx<0] += 1
        check[dy<0] += 1
        theta[check==2] = 3*(numpy.pi)/2. - numpy.arctan(dy[check==2]/dx[check==2])

        # if dx<0 and dy<0:
        #     theta = 3*(numpy.pi)/2. - numpy.arctan(dy/dx)
        
        return theta



    # def windProfileInformation(self, delta_xSep, delta_ySep):
    #     """Global parameters self.fittedlayerAlt, self.fittedgamma & fittedGSPOS used to define each integerHeights'
    #     wind direction and velocity.

    #     OUTPUTS:
    #     windVelocity - velocity of layer [m/s].
    #     windDirection - angle at which the layer is moving in 360 degree circle (0 at 12o'clock) [degrees]."""

    #     windDirection = numpy.zeros(self.nLayers)
    #     windVelocity = numpy.zeros(self.nLayers)
    #     for i in range(self.nLayers):
            
    #         relGsPos = numpy.array(([[0,0], [delta_ySep[i], delta_xSep[i]]]))
    #         relGamma = vectorAngle(relGsPos[0], relGsPos[1])
    #         print((relGamma * 180/numpy.pi))
    #         wd = 360 - ((relGamma * 180/numpy.pi) + 180)
    #         # print(wd)
    #         # windDirection[i] = wd
    #         windDirection[i] = (math.modf(wd/360.)[0])*360.

    #         #calculation of wind velocity - dependent on frequency and temporal offset
    #         displacement = numpy.sqrt(delta_xSep[i]**2 + delta_ySep[i]**2)
    #         windVelocity[i] = displacement*(self.wfsFreq/self.offsetStep)				 

    #     return windVelocity, windDirection