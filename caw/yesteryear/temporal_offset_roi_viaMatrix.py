def temporal_offset_roi(self):

	if self.l3s_wind==False:
		num_arrays = 1
		cents_array = numpy.zeros((num_arrays, self.shwfs_centroids.shape[0], self.shwfs_centroids.shape[1]))
		matrix_of_offsets = numpy.zeros((num_arrays, 2, 2*numpy.sum(self.n_subap), 2*numpy.sum(self.n_subap)))
	else:
		num_arrays = 2
		cents_array = numpy.zeros((num_arrays, self.shwfs_centroids.shape[0], self.shwfs_centroids.shape[1]))
		matrix_of_offsets = numpy.zeros((num_arrays, 2, 2*numpy.sum(self.n_subap), 2*numpy.sum(self.n_subap)))
		cents_array[1] = numpy.matmul(numpy.matmul(self.transform_matrix, self.shwfs_centroids.T).T, self.transform_matrix.T)
	cents_array[0] = self.shwfs_centroids
	

	width = int((2*self.wind_roi_envelope+1) * self.n_wfs)
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
			r = 2*self.n_subap[0]
			s = 2*2*self.n_subap[0]
			u = 0
			w = 2*self.n_subap[0]
			
			rr = 2*self.n_subap[0]
			ss = 2*2*self.n_subap[0]
			uu = 0
			ww = 2*self.n_subap[0]
			#loop over number of GS combinations
			
			for comb in range(self.combs):

				if rr==matrix_of_offsets.shape[2]:
					r += 2*self.n_subap[0]
					s += 2*self.n_subap[0]
					u += 2*self.n_subap[0]
					w += 2*self.n_subap[0]
					
					rr = r
					ss = s
					uu = u
					ww = w

				#induce positive and negative temporal slope offsets
				cents_pos_offset, cents_neg_offset = self.temp_offset_cents(cents_array[n_a], 
					dt, self.temporal_step, self.n_subap[self.selector[comb, 0]], self.n_subap[self.selector[comb, 1]], self.selector[comb])
				
				#Make covariance matrix -> map
				comb_matrix_pos = cross_cov(cents_pos_offset)[2*self.n_subap[self.selector[comb, 0]]:, :2*self.n_subap[self.selector[comb, 1]]]
				comb_matrix_neg = cross_cov(cents_neg_offset)[2*self.n_subap[self.selector[comb, 0]]:, :2*self.n_subap[self.selector[comb, 1]]] * self.mult_neg_offset
				matrix_of_offsets[n_a, 0, rr:ss, uu:ww] += comb_matrix_neg
				matrix_of_offsets[n_a, 1, rr:ss, uu:ww] += comb_matrix_pos

				rr += 2*self.n_subap[0]
				ss += 2*self.n_subap[0]

			for n_wfs in range(self.n_wfs):
				cents_auto_pos = cents_array[n_a][:self.shwfs_centroids.shape[0]-dt*self.temporal_step, n_wfs*2*self.n_subap[0]:(n_wfs+1)*2*self.n_subap[0]]
				cents_auto_neg = cents_array[n_a][dt*self.temporal_step:, n_wfs*2*self.n_subap[0]:(n_wfs+1)*2*self.n_subap[0]]
				auto_matrix_pos = cross_cov(cents_auto_pos)
				auto_matrix_neg = cross_cov(cents_auto_neg)

				matrix_of_offsets[n_a, 0, n_wfs*2*self.n_subap[0]: (n_wfs+1)*2*self.n_subap[0], n_wfs*2*self.n_subap[0]: (n_wfs+1)*2*self.n_subap[0]] = auto_matrix_neg
				matrix_of_offsets[n_a, 1, n_wfs*2*self.n_subap[0]: (n_wfs+1)*2*self.n_subap[0], n_wfs*2*self.n_subap[0]: (n_wfs+1)*2*self.n_subap[0]] = auto_matrix_pos

		matrix_of_offsets[n_a, 0] = self.mirror_covariance_matrix(matrix_of_offsets[n_a, 0], self.n_wfs, self.n_subap[0])
		matrix_of_offsets[n_a, 1] = self.mirror_covariance_matrix(matrix_of_offsets[n_a, 1], self.n_wfs, self.n_subap[0])

		if self.include_temp0==True:
			temp0_covMatrix = cross_cov(cents_array[n_a])
			matrix_of_offsets[n_a, 0] += temp0_covMatrix * self.mult_neg_offset
			matrix_of_offsets[n_a, 1] += temp0_covMatrix

		map_offset_neg = covMap_fromMatrix(matrix_of_offsets[n_a, 0], self.n_wfs, self.nx_subap, self.n_subap, self.pupil_mask, self.wind_map_axis, self.mm, self.mmc, self.md)
		map_offset_pos = covMap_fromMatrix(matrix_of_offsets[n_a, 1], self.n_wfs, self.nx_subap, self.n_subap, self.pupil_mask, self.wind_map_axis, self.mm, self.mmc, self.md)
		
		roi_neg = roi_from_map(map_offset_neg, self.gs_pos, self.pupil_mask, self.selector, self.wind_roi_belowGround, self.wind_roi_envelope)
		roi_pos = roi_from_map(map_offset_pos, self.gs_pos, self.pupil_mask, self.selector, self.wind_roi_belowGround, self.wind_roi_envelope)
		if self.zeroSep_cov==False:
			roi_neg[self.zeroSep_locations] = 0.
			roi_pos[self.zeroSep_locations] = 0.
			if self.separate_pos_neg_offsets==True:
				roi_offsets[n_a, :, :length] = roi_neg
				roi_offsets[n_a, :, length:] = roi_pos
			else:
				roi_offsets[n_a] += roi_neg + roi_pos

	pyplot.figure()
	# # pyplot.imshow(matrix_of_offsets[0,0])
	pyplot.imshow(roi_offsets[0])
	# pyplot.colorbar()
	# d=off

	return roi_offsets





def mirror_covariance_matrix(self, cov_mat, n_wfs, n_subaps):
	"""
	Mirrors a covariance matrix around the axis of the diagonal.

	Parameters:
		cov_mat (ndarray): The covariance matrix to mirror
		n_subaps (ndarray): Number of sub-aperture in each shwfs.

	Returns:
		(ndarray): Complete covariance matrix."""

	
	total_slopes = cov_mat.shape[0]

	n1 = 0
	for n in range(n_wfs):
		m1 = 0
		for m in range(n + 1):
			if n != m:
				n2 = n1 + 2 * n_subaps
				m2 = m1 + 2 * n_subaps

				cov_mat[m1: m2, n1: n2] = (
					numpy.swapaxes(cov_mat[n1: n2, m1: m2], 1, 0)
				)

				m1 += 2 * n_subaps
		n1 += 2 * n_subaps
	return cov_mat