



def tempOffsets(self, slopes, gsPos, windDirect, windL3S):

	slopes = numpy.matmul(numpy.matmul(self.transformMatrix, slopes.T).T, self.transformMatrix.T)

	matrix_of_offsets = numpy.zeros((2*self.nSubaps*self.nWfs, 2*self.nSubaps*self.nWfs))

	r = 2*self.nSubaps
	s = 2*2*self.nSubaps
	u = 0
	w = 2*self.nSubaps
	
	slice_of_offsets_aloft = 0.
	#loop over how many temporal offsets are being included in the fitting procedure
	for dt in range(1, 1+self.noOffsets):
		rr = 2*self.nSubaps
		ss = 2*2*self.nSubaps
		uu = 0
		ww = 2*self.nSubaps
		#loop over number of GS combinations
		
		for comb in range(self.combs):

			if rr==matrix_of_offsets.shape[0]:
				r += 2*self.nSubaps
				s += 2*self.nSubaps
				u += 2*self.nSubaps
				w += 2*self.nSubaps
				rr = r
				ss = s
				uu = u
				ww = w

			#induce positive and negative temporal slope offsets
			slopesOffsetPositive, slopesOffsetNegative = self.tempOffsetSlopes(slopes, dt, self.temporalStep, self.nSubaps, self.selector[comb], False)
			#Make covariance matrix -> map
			comb_matrices_posi = crossCov(slopesOffsetPositive)[2*self.nSubaps:, :2*self.nSubaps]
			# comb_matrices_nega = crossCov(slopesOffsetNegative)[2*self.nSubaps:, :2*self.nSubaps] * self.multNegativeTempOffset
			matrix_of_offsets[rr:ss, uu:ww] += comb_matrices_posi #+ comb_matrices_nega

			rr += 2*self.nSubaps
			ss += 2*self.nSubaps

		for n_wfs in range(self.nWfs):
			slopesAuto_posi = slopes[:slopes.shape[0]-dt*self.temporalStep, n_wfs*2*self.nSubaps:(n_wfs+1)*2*self.nSubaps]
			slopesAuto_nega = slopes[dt*self.temporalStep:, n_wfs*2*self.nSubaps:(n_wfs+1)*2*self.nSubaps]
			auto_matrices_posi = crossCov(slopesAuto_posi)
			auto_matrices_nega = crossCov(slopesAuto_nega) * self.multNegativeTempOffset

			matrix_of_offsets[n_wfs*2*self.nSubaps: (n_wfs+1)*2*self.nSubaps, n_wfs*2*self.nSubaps: (n_wfs+1)*2*self.nSubaps] = auto_matrices_posi + auto_matrices_nega

	matrix_of_offsets = self.mirror_covariance_matrix(matrix_of_offsets, self.nWfs, self.nSubaps)
	maps_of_offsets = covMapsFromMatrix(matrix_of_offsets, gsPos, self.nxSubaps, self.nSubaps, self.pupilMask, self.windSliceAxis, self.mm, self.mmc, self.md)
	slice_of_offsets = covSlicesFromMaps(maps_of_offsets, gsPos, self.pupilMask, self.selector, self.windBelowGround, self.windEnvelope)

	if self.includeTemp0==True:
		matrix_of_offsets += crossCov(slopes)
	if self.windL3S==True:
		matrix_of_offsets_aloft = numpy.matmul(numpy.matmul(self.transformMatrix, matrix_of_offsets), self.transformMatrix.T)
		maps_of_offsets_aloft = covMapsFromMatrix(matrix_of_offsets_aloft, gsPos, self.nxSubaps, self.nSubaps, self.pupilMask, self.windSliceAxis, self.mm, self.mmc, self.md)
		slice_of_offsets_aloft = covSlicesFromMaps(maps_of_offsets_aloft, gsPos, self.pupilMask, self.selector, self.windBelowGround, self.windEnvelope)

	# pyplot.figure()
	# pyplot.imshow(maps_of_offsets)

	pyplot.figure()
	pyplot.imshow(slice_of_offsets)
	# # pyplot.figure('l3s.1')
	# # pyplot.imshow(slice_of_offsets_aloft)

	d=off

	return slice_of_offsets, slice_of_offsets_aloft