### Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain). 
#   All rights reserved.
#   This code is distributed under the Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)).
#   Its use MUST also comply with the requirements of the individual licenses of all its dependencies.

### Load useful modules
import argparse, os, sys
import multiprocessing
import time
import numpy as np
from scipy.optimize import minimize
import nibabel as nib


def signal_gen(mri_seq,tissue_par):
	''' Function synthesising the MRI signals for the T2-IVIM-Kurtosis model

	    DEPENDENCIES: argparse, os, sys, multiprocessing, time (python standard library); numpy, scipy and nibabel		
		
	    INTERFACE
	    signal = signal_gen(mri_seq,tissue_par)
	    
	    PARAMETERS
	    - mri_seq: 2D matrix, with 1st row listing b-values (in s/mm^2) and 2nd row listing TE (in ms)
	    - tissue_par: list/array of tissue parameters, in the following order:
                           tissue_par[0] = s0 (apparent proton density)
                           tissue_par[1] = fv (vascular signal fraction, ranging in 0-1)                            
                           tissue_par[2] = Dv (apparent diffusion coefficient of vascular water, in um2/ms)
                           tissue_par[3] = T2v (T2 of vascular water, in ms)                      
                           tissue_par[4] = Dt (apparent diffusion coefficient of tissue, um2/ms)
                           tissue_par[5] = Kt (apparent excess kurtosis coefficient of tissue, dimensionless)
                           tissue_par[6] = T2t (T2 of tissue, in ms)     		
	    RETURNS
	    - signal: a numpy array of measurements generated according to the model
			
		         S = S0*(fv*exp(-b Dv - TE/T2v) + (1 - fv)*exp( -b Dt + (Kt/6)*(b Dt)^2 - TE/T2t)
	     
	    Author: Francesco Grussu, Vall d Hebron Institute of Oncology, Barcelona, Spain
		   <fgrussu@vhio.net> <francegrussu@gmail.com>'''
	

	### Handle inputs
	scheme = np.array(mri_seq,'float64')  # MRI sequence parameters	
	bvals = scheme[0,:]                   # b-values in s/mm2
	bvals = bvals/1000.0                  # b-values in ms/um2
	tes = scheme[1,:]                     # TE in ms
	s0_value = tissue_par[0]              # S0
	fv_value = tissue_par[1]              # fv
	Dv_value = tissue_par[2]              # Dv in um2/ms 
	T2v_value = tissue_par[3]             # T2v in ms
	Dt_value = tissue_par[4]              # Dt in um2/ms
	Kt_value = tissue_par[5]              # Kt
	T2t_value = tissue_par[6]             # T2t in ms

	### Calculate signal
	with np.errstate(divide='raise',invalid='raise'):
		try:

			signal = s0_value * ( fv_value*np.exp( -bvals*Dv_value -tes/T2v_value) + (1 - fv_value)*np.exp( -bvals*Dt_value + (1.0/6.0)*bvals*bvals*Dt_value*Dt_value*Kt_value -tes/T2t_value) )	

		except FloatingPointError:
			signal = 0.0 * bvals      # Just output zeros when tissue parameters do not make sense			

	### Output signal
	return signal
	


def Fobj(tissue_par,mri_seq,meas,sgmnoise,navg,parmin,parmax,wtikh):
	''' Fitting objective function to estimate T2-IVIM-Kurtosis tissue parameters		

	    DEPENDENCIES: argparse, os, sys, multiprocessing, time (python standard library); numpy, scipy and nibabel
		
	    INTERFACE
	    fobj = Fobj(tissue_par,mri_seq,meas,sgmnoise,navg,parmin,parmax,wtikh)

	    PARAMETERS
	    - tissue_par: list/array of tissue parameters, in the following order:
                           tissue_par[0] = s0 (apparent proton density)
                           tissue_par[1] = fv (vascular signal fraction, ranging in 0-1)                            
                           tissue_par[2] = Dv (apparent diffusion coefficient of vascular water, in s/mm2)
                           tissue_par[3] = T2v (T2 of vascular water, in ms)                      
                           tissue_par[4] = Dt (apparent diffusion coefficient of tissue, in s/mm2)
                           tissue_par[5] = Kt (apparent excess kurtosis coefficient of tissue, dimensionless)
                           tissue_par[6] = T2t (T2 of tissue, in ms)   
	    - mri_seq: 2D matrix, with 1st row listing b-values (in s/mm^2) and 2nd row listing TE (in ms)
	    - meas: list/array of measurements
	    - sgmnoise: noise standard deviation, to be used for offset Gaussian likelihood maximisation. 
	                Use sgmnoise = 0.0 for pure Gaussian noise.
	    - navg: number of signal averages for more accurate estimation of the noise floor from sgmnoise.
	            Use navg = 1 if signal averaging was not performed (NSA = 1 or NEX = 1, depending on
	            the vendor; note that if trace-DWI imaging was used, navg should be equal to 3). 
	            If sgmnoise = 0.0, navg has no effect
	    - parmin: array of lower bounds for tissue parameters (order: s0, fv, Dv, T2v, Dt, Kt, T2t); 
	             the true S0 bound will be obtained multiplyig parmin[0] by max(signal) 
	    - parmax: array of upper bounds for tissue parameters (order: s0, fv, Dv, T2v, Dt, Kt, T2t); 
	             the true S0 bound will be obtained multiplyig parmax[0] by max(signal) 
	    - wtikh: Tikhonov regularisation weight (must be >=0; set it to 0 for no regulsarisation.
	             Note that an effective regularisation weight value will depend on the range on which the MRI signal 
	             is defined: for instance, for an MRI signal varying in the range [0; SMAX], resonable regularisation weights 
	             are of the order of approx SMAX/3)
		
	    RETURNS
	    - fobj: objective function according to a offset Gaussian noise model with Tikhonov regularisation: 
	    
	    			fobj = F1 + w*F2 with
			
				F1 = SUM_OVER_n( (measurement_n - sqrt(prediction_n^2 + noise_floor^2))^2 )
				F2 = SUM_OVER_k( ( (p_k - pmin_k)/(pmax_k - pmin_k) )^2 )
				
		     Above:
		     -- n is the measurement index
		     -- k is the tissue parameter index
		     -- w >= 0 is the Tikhonov regularisation weight
		     -- signal predictions are obtained using the signal model implemented by function signal_gen()
		     -- noise_floor =  1.253*sgmnoise*np.sqrt(navg)
		     -- pmin_k is the lower bound for the k-th tissue parameter
		     -- pmax_k is the upper bound for the k-th tissue parameter
	     
	    Author: Francesco Grussu, Vall d Hebron Institute of Oncology,Barcelona, Spain
		   <fgrussu@vhio.net> <francegrussu@gmail.com>'''
	
	
	### Get noise floor
	if(sgmnoise!=0.0):	
		nfloor = np.sqrt(0.5*np.pi)*sgmnoise*np.sqrt(float(navg))
	else:
		nfloor = 0.0
	
	### Predict signals given tissue and sequence parameters
	pred = signal_gen(mri_seq,tissue_par)

	### Calculate objective function
	
	# Data fidelity term
	fobj = np.sum( ( np.array(meas) - np.sqrt(pred*pred + nfloor*nfloor) )**2 )
	
	# Optional regularisation term
	if(wtikh!=0):
		max_meas = np.max(meas)
		F2 = ( (tissue_par[0] - parmin[0]*max_meas)/(parmax[0]*max_meas - parmin[0]*max_meas) )**2
		for kk in range(1,tissue_par.size):
			F2 = F2 + ( (tissue_par[kk] - parmin[kk])/(parmax[kk] - parmin[kk]) )**2
		fobj = fobj + wtikh*F2
	
	### Output total objective function
	return fobj
	

	
def GridSearch(mri_seq,meas,sgmnoise,navg,parmin=[0.8,0.0,4.0,150.0,0.2,0.0,20.0],parmax=[12.0,1.0,100.0,250.0,3.0,5.0,140.0],nword=4,wtikh=0.0):
	''' Grid searchto initialise regularised non-linear least square fitting of the T2-IVIM-Kurtosis model

	    DEPENDENCIES: argparse, os, sys, multiprocessing, time (python standard library); numpy, scipy and nibabel
		
	    INTERFACE
	    tissue_estimate, fobj_grid = GridSearch(mri_seq,meas,nfloor,parmin=default_min,parmax=default_max,nword=4,wtikh=0.0)
	    
	    PARAMETERS
	    - mri_seq: 2D matrix, with 1st row listing b-values (in s/mm^2) and 2nd row listing TE (in ms)
	    - meas: list/array of measurements
	    - sgmnoise: noise standard deviation, to be used for offset Gaussian likelihood maximisation. 
	                Use sgmnoise = 0.0 for pure Gaussian noise.
	    - navg: number of signal averages for more accurate estimation of the noise floor from sgmnoise.
	            Use navg = 1 if signal averaging was not performed (NSA = 1 or NEX = 1, depending on
	            the vendor; note that if trace-DWI imaging was used, navg should be equal to 3). 
	            If sgmnoise = 0.0, navg has no effect
	    - parmin: lower bound for tissue parameters. Default parmin=default_min, with default_min
	              elements being (in order):
	                   S0 = 0.8 for apparent proton density (true S0 bound obtained multiplyig this number of max(signal))
                           fv = 0.0 for vascular signal fraction                           
                           Dv = 4 um2/ms for the vascular apparent diffusion coefficient
                           T2v = 150 ms for the vascular T2                    
                           Dt = 0.2 um2/ms for the tissue apparent diffusion coefficient
                           Kt = 0.0 for the tissue apparent excess kurtosis coefficient
                           T2t = 20.0 for the tissue T2
	    - parmax: upper bound for tissue parameters. Default parmax=default_max, with default_max
	              elements being (in order):
	                   S0 = 12.0 for apparent proton density (true S0 bound obtained multiplyig this number of max(signal))
                           fv = 1.0 for vascular signal fraction                        
                           Dv = 100 um2/ms for the vascular apparent diffusion coefficient
                           T2v = 250 ms for the vascular T2                    
                           Dt = 3.0 um2/ms for the tissue apparent diffusion coefficient
                           Kt = 5.0 for the tissue apparent excess kurtosis coefficient
                           T2t = 140.0 for the tissue T2	
            - nword: number of values to test for each tissue parameter in the grid search. Default nword=4
            - wtikh: Tikhonov regularisation weight (>=0). Default 0.0, implying no regularisation. 
	             Note that an effective regularisation weight value will depend on the range on which the MRI signal 
	             is defined: for instance, for an MRI signal varying in the range [0; SMAX], resonable regularisation weights 
	             are of the order of approx SMAX/3)
            	
	    RETURNS
	    - tissue_estimate: array of tissue parameter that minimise the objective function Fobj(), obtained from 
	                       a discrete list of tissue parameter candidates (i.e., the grid). Parameters are
                               tissue_estimate[0] = S0 (apparent proton density)
                               tissue_estimate[1] = fv (vascular signal fraction, ranging in 0-1)                            
                               tissue_estimate[2] = Dv (apparent diffusion coefficient of vascular water, in s/mm2)
                               tissue_estimate[3] = T2v (T2 of vascular water, in ms)                      
                               tissue_estimate[4] = Dt (apparent diffusion coefficient of tissue, in s/mm2)
                               tissue_estimate[5] = Kt (apparent excess kurtosis coefficient of tissue, dimensionless)
                               tissue_estimate[6] = T2t (T2 of tissue, in ms)   
	    - fobj_grid:       value of the minimised objective function  Fobj()
	     
	    Author: Francesco Grussu, Vall d Hebron Institute of Oncology, Barcelona, Spain
		   <fgrussu@vhio.net> <francegrussu@gmail.com>'''
	
	
	### Get noise floor
	if(sgmnoise!=0.0):	
		nfloor = np.sqrt(0.5*np.pi)*sgmnoise*np.sqrt(float(navg))
	else:
		nfloor = 0.0
	
	### Create grid
	s0min = 0.1*parmin[0]*np.max(meas)
	s0max = 0.9*parmax[0]*np.max(meas)
	fvmin = 0.1*parmin[1]
	fvmax = 0.9*parmax[1]
	Dvmin = 0.1*parmin[2]
	Dvmax = 0.9*parmax[2]
	T2vmin = 0.1*parmin[3]
	T2vmax = 0.9*parmax[3]
	Dtmin = 0.1*parmin[4]
	Dtmax = 0.9*parmax[4]
	Ktmin = 0.1*parmin[5]
	Ktmax = 0.9*parmax[5]
	T2tmin = 0.1*parmin[6]
	T2tmax = 0.9*parmax[6]
	s0vals = np.linspace(s0min,s0max,nword)	
	fvvals = np.linspace(fvmin,fvmax,nword)	
	Dvvals = np.linspace(Dvmin,Dvmax,nword)	
	T2vvals = np.linspace(T2vmin,T2vmax,nword)	
	Dtvals = np.linspace(Dtmin,Dtmax,nword)	
	Ktvals = np.linspace(Ktmin,Ktmax,nword)
	T2tvals = np.linspace(T2tmin,T2tmax,nword)
	s0arr,fvarr,Dvarr,T2varr,Dtarr,Ktarr,T2tarr = np.meshgrid(s0vals,fvvals,Dvvals,T2vvals,Dtvals,Ktvals,T2tvals)
	s0arr = s0arr.flatten()
	fvarr = fvarr.flatten()
	Dvarr = Dvarr.flatten()
	T2varr = T2varr.flatten()
	Dtarr = Dtarr.flatten()
	Ktarr = Ktarr.flatten()
	T2tarr = T2tarr.flatten()
	ngrid = s0arr.size         # Total number of elements in the grid
	
	## Calculate Tikhonov regularisation weight for all grid elements
	if(wtikh!=0):
		add0 = ( ( s0arr - parmin[0]*np.max(meas) )/( parmax[0]*np.max(meas) - parmin[0]*np.max(meas) ) )**2
		add1 = ( ( fvarr - parmin[1])/(parmax[1] - parmin[1]) )**2
		add2 = ( ( Dvarr - parmin[2])/(parmax[2] - parmin[2]) )**2
		add3 = ( ( T2varr - parmin[3])/(parmax[3] - parmin[3]) )**2
		add4 = ( ( Dtarr - parmin[4])/(parmax[4] - parmin[4]) )**2
		add5 = ( ( Ktarr - parmin[5])/(parmax[5] - parmin[5]) )**2
		add6 = ( ( T2tarr - parmin[6])/(parmax[6] - parmin[6]) )**2
		regarr = add0 + add1 + add2 + add3 + add4 + add5 + add6
		del add0, add1, add2, add3, add4, add5, add6
		
	### Generate signal predictions for each tissue parameter combination, biasing signals with the noise floor
	nmeas = mri_seq.shape[1]
	sigdict = np.zeros((ngrid,nmeas))
	for uu in range(0,ngrid):
	
		tissue_vals = np.array([ s0arr[uu], fvarr[uu], Dvarr[uu], T2varr[uu], Dtarr[uu], Ktarr[uu], T2tarr[uu] ])
		sigarray = signal_gen(mri_seq, tissue_vals)
		sigarray = np.sqrt(sigarray*sigarray + nfloor*nfloor)   # noise floor bias
		sigdict[uu,:] = sigarray
	
	### Perform the grid search
	measmat = np.tile(meas,(sigdict.shape[0],1))    # Matrix of actual MRI measurements
	if(wtikh!=0):
		fobj_array = nmeas*np.nanmean( (measmat - sigdict)**2 , axis=1) + wtikh*regarr
	else:
		fobj_array = nmeas*np.nanmean( (measmat - sigdict)**2 , axis=1) 
	min_idx = np.argmin(fobj_array)
	s0fit = s0arr[min_idx]
	fvfit = fvarr[min_idx]
	Dvfit = Dvarr[min_idx]
	T2vfit = T2varr[min_idx]
	Dtfit = Dtarr[min_idx]
	Ktfit = Ktarr[min_idx]
	T2tfit = T2tarr[min_idx]
	tissueout = np.array([s0fit,fvfit,Dvfit,T2vfit,Dtfit,Ktfit,T2tfit])
	Fobjfit = Fobj(tissueout,mri_seq,meas,sgmnoise,navg,parmin,parmax,wtikh)

	# Return	
	return tissueout, Fobjfit



def FitSlice(data):
	''' Fit T2-IVIM-Kurtosis one MRI slice stored as a 3D numpy array (2 image dimensions; 1 for multiple measurements)  
	    
	    DEPENDENCIES: argparse, os, sys, multiprocessing, time (python standard library); numpy, scipy and nibabel
	    
	    INTERFACE
	    data_out = FitSlice(data)
	     
	    PARAMETERS
	    - data: a list of 7 elements, such that
	            data[0] is a 3D numpy array contaning the data to fit. The first and second dimensions of data[0]
		            are the slice first and second dimensions, whereas the third dimension of data[0] stores
                            different MRI measurements
		    data[1] is a 2D numpy matrix, with 1st row listing b-values (in s/mm^2) and 2nd row listing TE (in ms)
		    data[2] is a 2D numpy array contaning the fitting mask within the MRI slice (see FitModel())
		    data[3] contains the lower bound for the tissue parameters. Please see signal_gen() and FitModel()
		    data[4] contains the upper bound for the tissue parameters. Please see signal_gen() and FitModel()
		    data[5] is a 2D numpy array contaning the noise standard deviation map within the MRI slice (see FitModel())
		    data[6] is the number of signal averages (NEX or NSA; note that trace-DW imaging has an intrinsic NEX = 3)
		    data[7] is the number of values to test for each tissue parameter in the grid search (see GridSearch())
		    data[8] is the Tikhonov regularisation weight (see FitModel())
		    data[9] is a scalar containing the index of the MRI slice in the 3D volume
	    
	    RETURNS
	    - data_out: a list of 8 elements, such that
                           data_out[0] = S0 estimates (apparent proton density)
                           data_out[1] = fv estimates (vascular signal fraction, ranging in 0-1)                            
                           data_out[2] = Dv estimates (apparent diffusion coefficient of vascular water, in s/mm2)
                           data_out[3] = T2v estimates (T2 of vascular water, in ms)                      
                           data_out[4] = Dt estimates (apparent diffusion coefficient of tissue, in s/mm2)
                           data_out[5] = Kt estimates (apparent excess kurtosis coefficient of tissue, dimensionless)
                           data_out[6] = T2t estimates (T2 of tissue, in ms)
                           data_out[7] = value of the minimised objective function
                           data_out[8] = fitting exit code
                           data_out[9] = index of the MRI slice in the 3D volume (same as data[3])
	
		    Fitted parameters in data_out will be stored as double-precision floating point (FLOAT64)
	     
	    Author: Francesco Grussu, Vall d Hebron Institute of Oncology, Barcelona, Spain
		   <fgrussu@vhio.net> <francegrussu@gmail.com>'''

	
	### Extract signals and sequence information from the input list
	signal_slice = data[0]       # Signal
	seq_value = data[1]          # Sequence
	mask_slice = data[2]         # fitting mask
	para_min = data[3]           # Lower bound for tissue parameters
	para_max = data[4]           # Upper bound for tissue parameters
	sigma_slice = data[5]        # noise standard deviation map
	nex = data[6]                # numper of signal averages
	nwgrid = data[7]             # Number of values to test for each tissue parameter in the grid search
	regweigh = data[8]           # Regularisation weight
	idx_slice = data[9]          # Slice index
	slicesize = signal_slice.shape       # Get number of voxels of current MRI slice along each dimension
	seq_value = np.array(seq_value)      # Make sure sequence parameters are stored as an array
 
	### Allocate output variables
	s0_slice = np.zeros(slicesize[0:2],'float64')
	fv_slice = np.zeros(slicesize[0:2],'float64')
	dv_slice = np.zeros(slicesize[0:2],'float64')
	t2v_slice = np.zeros(slicesize[0:2],'float64')
	dt_slice = np.zeros(slicesize[0:2],'float64')
	kt_slice = np.zeros(slicesize[0:2],'float64')
	t2t_slice = np.zeros(slicesize[0:2],'float64')
	exit_slice = np.zeros(slicesize[0:2],'float64')
	fobj_slice = np.zeros(slicesize[0:2],'float64')
	Nmeas = slicesize[2]   # Number of measurements


	### Fit monoexponential decay model in the voxels within the current slice
	for xx in range(0, slicesize[0]):
			for yy in range(0, slicesize[1]):
		
				# Get mask for current voxel
				mask_voxel = mask_slice[xx,yy]           # Fitting mask for current voxel

				# The voxel is not background: fit the signal model			
				if(mask_voxel==1):

					# Get signal and fitting mask
					sig_voxel = signal_slice[xx,yy,:]           # Extract signals for current voxel
					sig_voxel = np.array(sig_voxel)             # Convert signal to array
					max_sig = np.max(sig_voxel)            	   # Maximum signal intensity in this voxel
					sigma_voxel = sigma_slice[xx,yy]            # Noise standard deviation

					# Perform grid search to initialise non-linear least square fitting
					pmin = np.copy(para_min)
					pmax = np.copy(para_max)
					pmin[0] = pmin[0]*max_sig
					pmax[0] = pmax[0]*max_sig
					param_init, fobj_init = GridSearch(seq_value,sig_voxel,sigma_voxel,nex,parmin=pmin,parmax=pmax,nword=nwgrid,wtikh=regweigh)
	         
					# Perform non-linear least square fitting
					try:
						param_bound = ((pmin[0],pmax[0]),(pmin[1],pmax[1]),(pmin[2],pmax[2]),(pmin[3],pmax[3]),(pmin[4],pmax[4]),(pmin[5],pmax[5]),(pmin[6],pmax[6]),)
						modelfit = minimize(Fobj, param_init, method='L-BFGS-B', args=tuple([seq_value,sig_voxel,sigma_voxel,nex,pmin,pmax,regweigh]), bounds=param_bound)
						fit_exit = modelfit.success
						fobj_fit = modelfit.fun
						param_fit = modelfit.x
						s0_voxel = param_fit[0]
						fv_voxel = param_fit[1]
						dv_voxel = param_fit[2]
						t2v_voxel = param_fit[3]
						dt_voxel = param_fit[4]
						kt_voxel = param_fit[5]
						t2t_voxel = param_fit[6]
						fobj_voxel = fobj_fit
						if(fit_exit==True):
							exit_voxel = 1
						else:
							exit_voxel = -1
					except:
						s0_voxel = 0.0
						fv_voxel = 0.0
						dv_voxel = 0.0
						t2v_voxel = 0.0
						dt_voxel = 0.0
						kt_voxel = 0.0
						t2t_voxel = 0.0
						fobj_voxel = 0.0
						exit_voxel = -1
						
					
				# The voxel is background
				else:
					s0_voxel = 0.0
					fv_voxel = 0.0
					dv_voxel = 0.0
					t2v_voxel = 0.0
					dt_voxel = 0.0
					kt_voxel = 0.0
					t2t_voxel = 0.0
					fobj_voxel = 0.0
					exit_voxel = 0
				
				# Store fitting results for current voxel
				s0_slice[xx,yy] = s0_voxel
				fv_slice[xx,yy] = fv_voxel
				dv_slice[xx,yy] = dv_voxel
				t2v_slice[xx,yy] = t2v_voxel
				dt_slice[xx,yy] = dt_voxel
				kt_slice[xx,yy] = kt_voxel
				t2t_slice[xx,yy] = t2t_voxel
				fobj_slice[xx,yy] = fobj_voxel
				exit_slice[xx,yy] = exit_voxel


	### Create output list storing the fitted parameters and then return
	data_out = [s0_slice, fv_slice, dv_slice, t2v_slice, dt_slice, kt_slice, t2t_slice, fobj_slice, exit_slice, idx_slice]
	return data_out
	



def FitModel(sig_nifti,seq_text,output_rootname,ncpu=1,mask_nifti=None,sigma_nifti=None,nex=1,pMin=[0.8,0.0,4.0,150.0,0.2,0.0,20.0],pMax=[12.0,1.0,100.0,250.0,3.0,5.0,140.0],gridres=3,wtk=0.0):
	''' Fit the T2-IVIM-Kurtosis model s(b,TE) = s0*(fv*exp(-bDv -TE/T2v) + (1-fv)*exp(-bDt +(Kt/6)*b^2*Dt^2 -TE/T2t ))
	    
	    DEPENDENCIES: argparse, os, sys, multiprocessing, time (python standard library); numpy, scipy and nibabel
            
	    INTERFACES
	    FitModel(sig_nifti,seq_text,output_rootname,ncpu=1,mask_nifti=None,sigma_nifti=None,nex=1,pMin=defmin,pMax=defmax,gridres=4,wtk=0.0)
	     
	    PARAMETERS
	    - sig_nifti: path of a Nifti file storing the multi b-value data as 4D data.
	    - seq_text: path of a text file storing the b-values (s/mm^2) used to acquire the data.
	    - output_rootname: base name of output files. Output files will end in 
                               "_s0.nii"      --> apparent proton density
                               "_fv.nii"      --> vascular signal fraction
                               "_dv.nii"      --> apparent vascular diffusion coefficient (um2/ms)
                               "_t2v.nii"     --> vascular T2 (ms)
                               "_dt.nii"      --> apparent tissue diffusion coefficient (um2/ms)
                               "_kt.nii"      --> apparent tissue excess kurtosis coefficient
                               "_t2t.nii"     --> tissue T2 (ms)
                               "_fobj.nii"    --> fitting objective function
                               "_exit.nii"    --> fitting exit code (1: success; -1: warning; 0: background)
                               
			    Output files will be stored as double-precision floating point (FLOAT64)
			
	    - ncpu: number of processors to be used for computation (optional; default: 1)
	    - mask_nifti: path of a Nifti file storing a binary mask, where 1 flgas voxels where the 
			  signal model needs to be fitted, and 0 otherwise (optional; default: None, i.e. no mask)
	    - sigma_nifti: path of a Nifti file storing voxel-wise estimates of the noise standard deviation, which
	                   will be used to calculate the noise floor for offset-Gaussian likelihood maximisation
	                   (optional; default: None, i.e. standard Gaussian noise)
	    - nex: number of signal averages (NSA), also known as number of excitations (NEX) (optional, and relevant 
	           only if a noise standard deviation map is provided via sigma_nifti; default: 1). 
	           Note that trace-DW imaging has an intrinsic number of signal averages of 3, since images for 
	           three mutually orthogonal gradients are acquired and averaged.
	    - pMin: lower bound for tissue parameters. Default pMin=defmin, with defmin
	            elements being (in order): 
	                   S0 = 0.8 for apparent proton density (true S0 bound obtained multiplyig this number of max(signal))
                           fv = 0.0 for vascular signal fraction                           
                           Dv = 4 um2/ms for the vascular apparent diffusion coefficient
                           T2v = 150 ms for the vascular T2                    
                           Dt = 0.2 um2/ms for the tissue apparent diffusion coefficient
                           Kt = 0.0 for the tissue apparent excess kurtosis coefficient
                           T2t = 20.0 for the tissue T2
	    - pMax: upper bound for tissue parameters. Default pMax=defmax, with defmax
	            elements being (in order):
	                   S0 = 12.0 for apparent proton density (true S0 bound obtained multiplyig this number of max(signal))
                           fv = 1.0 for vascular signal fraction                        
                           Dv = 100 um2/ms for the vascular apparent diffusion coefficient
                           T2v = 250 ms for the vascular T2                    
                           Dt = 3.0 um2/ms for the tissue apparent diffusion coefficient
                           Kt = 5.0 for the tissue apparent excess kurtosis coefficient
                           T2t = 140.0 for the tissue T2	
	    - gridres: grid depth for grid search (i.e., number of test values for each tissue parameter; 
	               default gridres = 3)
	    - wtk: Tikhonov regularisation weight (must be <=0; default 0.0, implying no regularisation. 
	           Note that an effective regularisation weight value will depend on the range on which the MRI signal 
	           is defined: for instance, for an MRI signal varying in the range [0; SMAX], resonable regularisation weights 
	           are of the order of approx SMAX/3)
	                
	    Author: Francesco Grussu, Vall d Hebron Institute of Oncology, Barcelona, Spain
		   <fgrussu@vhio.net> <francegrussu@gmail.com>'''
	
	
	### Get time
	timeinitial = time.time()	

	### Check number of threads for model fitting
	ncpu_physical = multiprocessing.cpu_count()
	if ncpu>ncpu_physical:
		print('')
		print('WARNING: {} CPUs were requested. Using {} instead (all available CPUs)...'.format(ncpu,ncpu_physical))			 
		print('')
		ncpu = ncpu_physical     # Do not open more workers than the physical number of CPUs



	### Load MRI data
	print('    ... loading input data')
	
	# Make sure MRI data exists
	try:
		sig_obj = nib.load(sig_nifti)
	except:
		print('')
		raise RuntimeError('ERROR: the 4D input NIFTI file {} does not exist or is not in NIFTI format.'.format(me_nifti))			 
	
	# Get image dimensions and convert to float64
	sig_data = sig_obj.get_fdata()
	imgsize = sig_data.shape
	sig_data = np.array(sig_data,'float64')
	imgsize = np.array(imgsize)
	sig_header = sig_obj.header
	sig_affine = sig_header.get_best_affine()
	sig_dims = sig_obj.shape
	
	# Make sure that the text file with sequence parameters exists and makes sense
	try:
		seqarray = np.loadtxt(seq_text)
		seqarray = np.array(seqarray,'float64')
		seqarray_size = seqarray.shape[1]
	except:
		print('')
		raise RuntimeError('ERROR: the b-value file {} does not exist or is not a numeric text file.'.format(seq_text))			 

			
	# Check consistency of sequence parameter file and number of measurements
	if imgsize.size!=4:
		print('')
		raise RuntimeError('ERROR: the input file {} is not a 4D nifti.'.format(sig_nifti))					 

	if seqarray_size!=imgsize[3]:
		print('')
		raise RuntimeError('ERROR: the number of measurements in {} does not match the sequence parameter text file {}.'.format(sig_nifti,seq_text))

	seq = seqarray

	### Deal with optional arguments: mask
	if mask_nifti is not None:
		try:
			mask_obj = nib.load(mask_nifti)
		except:
			print('')
			raise RuntimeError('ERROR: the mask file {} does not exist or is not in NIFTI format.'.format(mask_nifti))
		
		# Make sure that the mask has consistent header with the input data containing the measurements
		mask_dims = mask_obj.shape		
		mask_header = mask_obj.header
		mask_affine = mask_header.get_best_affine()	
		
		# Make sure the mask is a 3D file
		mask_data = mask_obj.get_fdata()
		masksize = mask_data.shape
		masksize = np.array(masksize)
		if masksize.size!=3:
			print('')
			print('WARNING: the mask file {} is not a 3D Nifti file. Ignoring mask...'.format(mask_nifti))				 
			print('')
			mask_data = np.ones(imgsize[0:3],'float64')
		elif ( (np.sum(sig_affine==mask_affine)!=16) or (sig_dims[0]!=mask_dims[0]) or (sig_dims[1]!=mask_dims[1]) or (sig_dims[2]!=mask_dims[2]) ):
			print('')
			print('WARNING: the geometry of the mask file {} does not match that of the input data. Ignoring mask...'.format(mask_nifti))
			print('')
			mask_data = np.ones(imgsize[0:3],'float64')
		else:
			mask_data = np.array(mask_data,'float64')
			# Make sure mask data is a numpy array
			mask_data[mask_data>0] = 1
			mask_data[mask_data<=0] = 0

	else:
		mask_data = np.ones(imgsize[0:3],'float64')
		
		
	### Deal with optional arguments: noise standard deviation map
	if sigma_nifti is not None:
		try:
			sigma_obj = nib.load(sigma_nifti)
		except:
			print('')
			raise RuntimeError('ERROR: the noise map file {} does not exist or is not in NIFTI format.'.format(sigma_nifti))

		# Make sure that the noise map has consistent header with the input data containing the measurements
		sigma_dims = sigma_obj.shape		
		sigma_header = sigma_obj.header
		sigma_affine = sigma_header.get_best_affine()	
		
		# Make sure the mask is a 3D file
		sigma_data = sigma_obj.get_fdata()
		sigmasize = sigma_data.shape
		sigmasize = np.array(sigmasize)
		if sigmasize.size!=3:
			print('')
			print('WARNING: the noise map file {} is not a 3D Nifti file. Ignoring noise map...'.format(sigma_nifti))				 
			print('')
			sigma_data = np.zeros(imgsize[0:3],'float64')     # Invalid sigma map, ignore it - flag this with 0 everywhere, as noise floor bias is proprtional to sigma
		elif ( (np.sum(sig_affine==sigma_affine)!=16) or (sig_dims[0]!=sigma_dims[0]) or (sig_dims[1]!=sigma_dims[1]) or (sig_dims[2]!=sigma_dims[2]) ):
			print('')
			print('WARNING: the geometry of the noise map file {} does not match that of the input data. Ignoring noise map...'.format(sigma_nifti))
			print('')
			sigma_data = np.zeros(imgsize[0:3],'float64')     # Invalid sigma map, ignore it - flag this with 0 everywhere, as noise floor bias is proprtional to sigma
		else:
			sigma_data = np.array(sigma_data,'float64')
	else:
		sigma_data = np.zeros(imgsize[0:3],'float64')     # No mask was provided - flag this with 0 everywhere, as noise floor bias is proprtional to sigma
				

	### Allocate memory for outputs
	s0_data = np.zeros(imgsize[0:3],'float64')	       # (T1-weighted) apparent proton density
	fv_data = np.zeros(imgsize[0:3],'float64')	       # Vascular signal fraction
	dv_data = np.zeros(imgsize[0:3],'float64')	       # Apparent vascular diffusivity
	t2v_data = np.zeros(imgsize[0:3],'float64')	       # Vascular T2
	dt_data = np.zeros(imgsize[0:3],'float64')	       # Apparent tissue diffusivity
	kt_data = np.zeros(imgsize[0:3],'float64')	       # Apparent tissue kurtosis
	t2t_data = np.zeros(imgsize[0:3],'float64')	       # Tissue T2
	fobj_data = np.zeros(imgsize[0:3],'float64')	       # Fitting objective function
	exit_data = np.zeros(imgsize[0:3],'float64')	       # Exit code
	


	#### Fitting
	print('    ... model fitting')
	
	# Create the list of input data
	inputlist = [] 
	for zz in range(0, imgsize[2]):
		sliceinfo = [sig_data[:,:,zz,:],seq,mask_data[:,:,zz],pMin,pMax,sigma_data[:,:,zz],nex,gridres,wtk,zz]  # List of information relative to the zz-th MRI slice
		inputlist.append(sliceinfo)     # Append each slice list and create a longer list of MRI slices whose processing will run in parallel
	
	# Call a pool of workers to run the fitting in parallel if parallel processing is required (and if the the number of slices is > 1)
	if ncpu>1 and imgsize[2]>1:

		# Create the parallel pool and give jobs to the workers
		fitpool = multiprocessing.Pool(processes=ncpu)  # Create parallel processes
		fitpool_pids_initial = [proc.pid for proc in fitpool._pool]  # Get initial process identifications (PIDs)
		fitresults = fitpool.map_async(FitSlice,inputlist)      # Give jobs to the parallel processes
		
		# Busy-waiting: until work is done, check whether any worker dies (in that case, PIDs would change!)
		while not fitresults.ready():
			fitpool_pids_new = [proc.pid for proc in fitpool._pool]  # Get process IDs again
			if fitpool_pids_new!=fitpool_pids_initial:               # Check whether the IDs have changed from the initial values
				print('')					 # Yes, they changed: at least one worker has died! Exit with error
				raise RuntimeError('ERROR: some processes died during parallel fitting.')					 
		
		# Work done: get results
		fitlist = fitresults.get()

		# Collect fitting output and re-assemble MRI slices		
		for kk in range(0, imgsize[2]):					
			fitslice = fitlist[kk]    # Fitting output relative to kk-th element in the list
			slicepos = fitslice[9]    # Spatial position of kk-th MRI slice
			s0_data[:,:,slicepos] = fitslice[0]        # Parameter S0
			fv_data[:,:,slicepos] = fitslice[1]        # Parameter fv 
			dv_data[:,:,slicepos] = fitslice[2]        # Parameter Dv
			t2v_data[:,:,slicepos] = fitslice[3]       # Parameter T2v
			dt_data[:,:,slicepos] = fitslice[4]        # Parameter Dt
			kt_data[:,:,slicepos] = fitslice[5]        # Parameter Kt
			t2t_data[:,:,slicepos] = fitslice[6]       # Parameter T2t
			fobj_data[:,:,slicepos] = fitslice[7]      # Fitting objective funtcion
			exit_data[:,:,slicepos] = fitslice[8]      # Fitting exit code

	# Run serial fitting as no parallel processing is required (it can be very long)
	else:
		for kk in range(0, imgsize[2]):
			fitslice = FitSlice(inputlist[kk])   # Fitting output relative to kk-th element in the list
			slicepos = fitslice[9]    # Spatial position of kk-th MRI slice
			s0_data[:,:,slicepos] = fitslice[0]        # Parameter S0
			fv_data[:,:,slicepos] = fitslice[1]        # Parameter fv 
			dv_data[:,:,slicepos] = fitslice[2]        # Parameter Dv
			t2v_data[:,:,slicepos] = fitslice[3]       # Parameter T2v
			dt_data[:,:,slicepos] = fitslice[4]        # Parameter Dt
			kt_data[:,:,slicepos] = fitslice[5]        # Parameter Kt
			t2t_data[:,:,slicepos] = fitslice[6]       # Parameter T2t
			fobj_data[:,:,slicepos] = fitslice[7]      # Fitting objective funtcion
			exit_data[:,:,slicepos] = fitslice[8]      # Fitting exit code


	### Remove NaNs and Infs
	exit_data[np.isnan(s0_data)] = -1
	exit_data[np.isinf(s0_data)] = -1
	exit_data[np.isnan(fv_data)] = -1
	exit_data[np.isinf(fv_data)] = -1
	exit_data[np.isnan(dv_data)] = -1
	exit_data[np.isinf(dv_data)] = -1
	exit_data[np.isnan(t2v_data)] = -1
	exit_data[np.isinf(t2v_data)] = -1
	exit_data[np.isnan(dt_data)] = -1
	exit_data[np.isinf(dt_data)] = -1
	exit_data[np.isnan(kt_data)] = -1
	exit_data[np.isinf(kt_data)] = -1
	exit_data[np.isnan(t2t_data)] = -1
	exit_data[np.isinf(t2t_data)] = -1
	exit_data[np.isnan(fobj_data)] = -1
	exit_data[np.isinf(fobj_data)] = -1
	s0_data[np.isnan(s0_data)] = 0.0
	s0_data[np.isinf(s0_data)] = 0.0	
	fv_data[np.isnan(fv_data)] = 0.0
	fv_data[np.isinf(fv_data)] = 0.0	
	dv_data[np.isnan(dv_data)] = 0.0
	dv_data[np.isinf(dv_data)] = 0.0
	t2v_data[np.isnan(t2v_data)] = 0.0
	t2v_data[np.isinf(t2v_data)] = 0.0
	dt_data[np.isnan(dt_data)] = 0.0
	dt_data[np.isinf(dt_data)] = 0.0
	kt_data[np.isnan(kt_data)] = 0.0
	kt_data[np.isinf(kt_data)] = 0.0
	t2t_data[np.isnan(t2t_data)] = 0.0
	t2t_data[np.isinf(t2t_data)] = 0.0	
	fobj_data[np.isnan(fobj_data)] = 0.0
	fobj_data[np.isnan(fobj_data)] = 0.0
	
	### Save the output maps
	print('    ... saving output files')
	buffer_header = sig_obj.header
	buffer_header.set_data_dtype('float64')   # Make sure we save output NIFTIs as float64, even if input header indicates a different data type

	s0_obj = nib.Nifti1Image(s0_data,sig_obj.affine,buffer_header) 
	nib.save(s0_obj, '{}_s0.nii'.format(output_rootname))

	fv_obj = nib.Nifti1Image(fv_data,sig_obj.affine,buffer_header) 
	nib.save(fv_obj, '{}_fv.nii'.format(output_rootname))
	
	dv_obj = nib.Nifti1Image(dv_data,sig_obj.affine,buffer_header) 
	nib.save(dv_obj, '{}_dv.nii'.format(output_rootname))

	t2v_obj = nib.Nifti1Image(t2v_data,sig_obj.affine,buffer_header) 
	nib.save(t2v_obj, '{}_t2v.nii'.format(output_rootname))
	
	dt_obj = nib.Nifti1Image(dt_data,sig_obj.affine,buffer_header) 
	nib.save(dt_obj, '{}_dt.nii'.format(output_rootname))
	
	kt_obj = nib.Nifti1Image(kt_data,sig_obj.affine,buffer_header) 
	nib.save(kt_obj, '{}_kt.nii'.format(output_rootname))
	
	t2t_obj = nib.Nifti1Image(t2t_data,sig_obj.affine,buffer_header) 
	nib.save(t2t_obj, '{}_t2t.nii'.format(output_rootname))
	
	fobj_obj = nib.Nifti1Image(fobj_data,sig_obj.affine,buffer_header) 
	nib.save(fobj_obj, '{}_fobj.nii'.format(output_rootname))
	
	exit_obj = nib.Nifti1Image(exit_data,sig_obj.affine,buffer_header) 
	nib.save(exit_obj, '{}_exit.nii'.format(output_rootname))
	
	if sigma_nifti is not None:
		nfloor_data = np.sqrt(0.5*np.pi)*np.sqrt(float(nex))*sigma_data*mask_data
		nfloor_obj = nib.Nifti1Image(nfloor_data,sig_obj.affine,buffer_header) 
		nib.save(nfloor_obj, '{}_nfloor.nii'.format(output_rootname))
	
	### Done
	timefinal = time.time()
	print('    ... done - it took {} sec'.format(timefinal - timeinitial))
	print('')




# Run the module as a script when required
if __name__ == "__main__":

	
	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='Non-linear least square fitting with offset-Gaussian noise model and Tikhonov regularisation of the two-pool T2-IVIM-Kurtosis model to diffusion MRI images obtained by varying b-value b and echo time TE. This provides voxel-wise stimates of s0 (apparent proton density), fv (vascular signal fraction), Dv (vascular apparent diffusion coefficient), T2v (vascular T2), Dt (tissue apparent diffusion coefficient), Kt (tissue apparent excess kurtosis coefficient), T2t (tissue T2), so that the signal model is s(b,TE) = s0*(fv*exp(-bDv -TE/T2v) + (1-fv)*exp(-bDt +(Kt/6)*b^2*Dt^2 -TE/T2t )). Note that this code can be used even if the TE was not varied during acquisition: in this latter case, a very short "fake" TE value (e.g., 1ms) should be indicated for all MRI measurements, and both T2v and T2t should be fixed to a very long "fake" value (e.g., 1000ms) using options --pmin and --pmax (see below). Dependencies: argparse, os, sys, multiprocessing, time (python standard library); numpy, scipy and nibabel. Author: Francesco Grussu, Vall d Hebron Institute of Oncology, Barcelona, Spain (<fgrussu@vhio.net> <francegrussu@gmail.com>)')
	parser.add_argument('dwi', help='4D Nifti file of magnitude images from a diffusion MRI experiment obtained by varying b-value b and echo time TE.')
	parser.add_argument('scheme', help='text file storing b-values and echo times used to acquired the images (space-separated elements; 1st row: b-values in s/mm^2; 2nd row: echo times TE in ms). If all images were acquired with the same echo time (so that vascular/tissue T2 cannot be modelled), enter a very short TE for all measurement (e.g., 1 ms) and fix both vascular and tissue T2 to a very long value (for instance, 1000 ms) using the --pmin and --pmax options below.')
	parser.add_argument('out', help='root name for output files, to which the following termination strings will be added: "*_s0.nii" for the apparent proton density map; "*_fv.nii" for the vascular signal fraction map; "*_dv.nii" for the vascular apparent diffusion coefficient map (in um2/ms); "*_t2v.nii" for the vascular T2 map (in ms); "*_dt.nii" for the tissue apparent diffusion coefficien map (in um2/ms); "*_kt.nii" for the tissue apparent excess kurtosis map; "*_t2t.nii" for the tissue T2 (in ms); "*_fobj.nii" for the fitting objective function; "*_exit.nii" for the fitting exit code (1: success; -1: warning; 0: background); "*_nfloor.nii" for the noise floor map (if a noise map sigma was provided with option --sigma)')
	parser.add_argument('--mask', metavar='<file>', help='3D Nifti storing the tissue mask, so that voxels storing 1 indicate that fitting is required, while voxels storing 0 indicate that fitting is not required')
	parser.add_argument('--sigma', metavar='<file>', help='3D Nifti storing a voxel-wise noise standard deviation map, which will be used to model the noise floor in the offset-Gaussian fitting objective function')
	parser.add_argument('--nsa', metavar='<value>', default='1', help='number of signal averages (optional, and relevant only if a noise standard deviation map is provided with option --sigma; default: 1. Note that in some vendors it is known as NSA or NEX, i.e. number of excitations. Remember that trace-DW imaging has an intrinsic number of signal averages of 3, since images for three mutually orthogonal gradients are acquired and averaged)')
	parser.add_argument('--ncpu', metavar='<value>', help='number of threads to be used for computation (default: all available threads)')
	parser.add_argument('--grid', metavar='<value>', default='3', help='number of values for each tissue parameter in grid search (default: 3; be aware that computation time grows considerably as this parameter increases)')
	parser.add_argument('--pmin', metavar='<list>', help='comma-separaterd list storing the lower bounds for tissue parameters s0, fv, Dv (um2/ms), T2v (ms), Dt (um2/ms), Kt, T2t (ms). Example: --pmin 0.6,0.1,10.0,200.0,0.3,-0.5,50.0 (default: 0.8,0.0,4.0,150.0,0.2,0.0,20.0). Note: the true s0 bound will be obtained multiplying the value passed here by max(signal).')
	parser.add_argument('--pmax', metavar='<list>', help='comma-separaterd list storing the upper bounds for tissue parameters s0, fv, Dv (um2/ms), T2v (ms), Dt (um2/ms), Kt, T2t (ms). Example: --pmax 3.5,1.0,80.0,600.0,2.8,3.0,150.0 (default: 12.0,1.0,100.0,250.0,3.0,5.0,140.0). Note: the true s0 bound will be obtained multiplying the value passed here by max(signal).')
	parser.add_argument('--wtk', metavar='<value>', default='0.0', help='Tikhonov regularisation weight used in the fitting objective function (default 0.0; if set to 0.0, no regularisation is performed. Note that an effective regularisation weight value will depend on the range on which the MRI signal is defined: for instance, for an MRI signal varying in the range [0; SMAX], resonable regularisation weights are of the order of approx SMAX/3)')
	args = parser.parse_args()   

	### Get input arguments
	sigfile = args.dwi       # DWI scan
	seqfile = args.scheme    # Sequence parameters
	outroot = args.out       # Output
	maskfile = args.mask     # Mask
	sigmafile = args.sigma   # Noise standard deviation
	nsa = int(args.nsa)      # Number of signal averages
	ngrid = int(args.grid)   # Grid search depth
	treg = float(args.wtk)   # Tikhonov regularisation weight
	if treg<0.0:
		print('')
		raise RuntimeError('ERROR: the regularisation weight {} must be equal to or grater than zero.'.format(args.wtk))
		
	# Number of threads
	nprocess = args.ncpu	# Number of threads for parallel fitting
	if isinstance(nprocess, str)==1:
		# A precise number of CPUs has been requested
		nprocess = int(float(nprocess))
	else:
		# No precise number of CPUs has been requested: use all available threads
		nprocess = multiprocessing.cpu_count()
	
	# Lower bounds
	if args.pmin is not None:
		pmin = (args.pmin).split(',')
		pmin = np.array( list(map( float,pmin )) )
		pmin_str = args.pmin
	else:
		pmin = np.array([0.8,0.0,4.0,150.0,0.2,0.0,20.0])
		pmin_str = "0.8,0.0,4.0,150.0,0.2,0.0,20.0"

	
	# Upper bounds
	if args.pmax is not None:
		pmax = (args.pmax).split(',')
		pmax = np.array( list(map( float,pmax )) )
		pmax_str = args.pmax
	else:
		pmax = np.array([12.0,1.0,100.0,250.0,3.0,5.0,140.0])
		pmax_str = "12.0,1.0,100.0,250.0,3.0,5.0,140.0"	


	## Print information
	print('')
	print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
	print('                         getT2ivimkurtReg.py                          ')
	print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
	print('')
	print('* Input NIFTI file storing the DWI scan: {}'.format(sigfile))
	print('* Input sequence parameter file: {}'.format(seqfile))
	print('* Output root file name: {}'.format(outroot))
	if args.mask is not None:
		print('* NIFTI file storing tissue mask: {}'.format(args.mask))
	print('* Tissue parameter lower bounds: {}'.format(pmin))
	print('* Tissue parameter upper bounds: {}'.format(pmax))
	print('* Grid search depth: {}'.format(ngrid))
	if args.sigma is not None:
		print('* NIFTI file storing noise standard deviation map: {}'.format(args.sigma)) 
		print('* Number of signal averages: {}'.format(nsa))
	print('* Number of threads for model fitting: {}'.format(nprocess))
	print('* Tikhonov regularisation weight: {}'.format(treg))
	print('')

	## Run fitting
	FitModel(sigfile, seqfile, outroot, ncpu=nprocess, mask_nifti=maskfile, sigma_nifti=sigmafile, nex=nsa, pMin=pmin, pMax=pmax, gridres=ngrid, wtk=treg)
	
	### Done
	sys.exit(0)


