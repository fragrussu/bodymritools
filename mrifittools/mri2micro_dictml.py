### Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain). 
#   All rights reserved.
#   This code is distributed under the Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)).
#   Its use MUST also comply with the requirements of the individual licenses of all its dependencies.

import argparse, os, sys
from configparser import Interpolation
import nibabel as nib
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import LinearNDInterpolator
import multiprocessing as mp
import time



def _train_rbf(pars,sigs,smooth,poldeg):
	
	dict_too_big = True
	cnt = 0
	dict_size = np.linspace(1,0.05,20)
	
	# Start using the full dictionary, and then subsample it in case it gives memory error with RBF interpolation
	np.random.seed(19870210)
	while(dict_too_big):
		
		print('        (using {} % of the dictionary)'.format(dict_size[cnt]*100.0))
		try:	
			nmicro = pars.shape[0]
			nsmall = int(np.round(dict_size[cnt]*nmicro))
			if(nsmall==nmicro):
				myrbf = RBFInterpolator(pars, sigs, smoothing=smooth, degree=poldeg)
			else:
				idx = np.random.choice(nmicro,size=nsmall,replace=False)
				myrbf = RBFInterpolator(pars[idx,:], sigs[idx,:], smoothing=smooth, degree=poldeg)
			dict_too_big = False
		
		except:
			cnt = cnt + 1
			if(cnt==dict_size.size):
				raise RuntimeError('ERROR. Your dictionary may be too big. Try using fewer than {} % of its total entries.'.format(100.0*dict_size[-1]))
	
	# Return RBF interpolator 	
	return myrbf

	
def _train_linearND(pars,sigs,scaleflag):
	mylinearND = LinearNDInterpolator(pars, sigs, rescale=scaleflag)
	return mylinearND


def _sig(tispar,regmodel,regressor):

	
	# Get MRI signals using a regressor trained on numerical data, implementing the forward model signal = f(tissue parameters) for the given MRI protocol
	if(regmodel=='rbf'):                 # Radial basis function interpolation
		mrisig = regressor([tispar]) # NOTE: regressor([tispar]) is needed rather than regressor(tispar) as a 2D input array is expected (size 1xNmeas)
		mrisig = np.squeeze(mrisig)  # For the same reason, we also need a call to mrisig = np.squeeze(mrisig) - i.e., to make mrisig back as a 1D array
	elif(regmodel=='linearND'):          # Piece-wise linear interpolation in N dimensions 
		mrisig = regressor([tispar]) # NOTE: regressor([tispar]) is needed rather than regressor(tispar) as a 2D input array is expected (size 1xNmeas)
		mrisig = np.squeeze(mrisig)  # For the same reason, we also need a call to mrisig = np.squeeze(mrisig) - i.e., to make mrisig back as a 1D array
	else:
		raise RuntimeError('ERROR: unkown regression algorithm {}'.format(regmodel))
	
	
	# Return MRI signals
	return mrisig


def _fobj(tisp,meas,sgm,navg,rgmod,rgobj,lnorm,lam,pmin,pmax):

	# Synthesise MRI signal
	measpred = _sig(tisp,rgmod,rgobj)

	# Get noise floor
	nf = sgm*np.sqrt(0.5*np.pi)*np.sqrt(float(navg))
	
	# Calculate data fidelity term of the objective function (MSE with offset noise floor model)
	fdata = np.nanmean( ( meas - np.sqrt(measpred**2 + nf**2) )**2 )

	# Calculate regularisation term of the objective function (L1 or L2 norm)
	freg = 0.0
	npar = tisp.size
	if(lnorm==1):
		for pp in range(0,npar):
			freg = freg + np.abs( 0.5 - (tisp[pp] - pmin[pp])/(pmax[pp] - pmin[pp]) )
	elif(lnorm==2):
		for pp in range(0,npar):
			freg = freg + ( 0.5 - (tisp[pp] - pmin[pp])/(pmax[pp] - pmin[pp]) )*( 0.5 - (tisp[pp] - pmin[pp])/(pmax[pp] - pmin[pp]) )
	else:
		raise RuntimeError('ERROR. An L{}-norm is not supported for objective function regularisation'.format(lnorm))
	freg = freg/float(npar)

	# Calculate total objective function
	fout = (1.0 - lam)*fdata + lam*freg

	# Return objective function
	return fout




def _procslice(inlist):

	# Input information for current MRI slice
	navg = float(inlist[0])
	dict_pars = inlist[1]
	dict_sigs = inlist[2]
	k_data = inlist[3]
	m_data = inlist[4]
	sgm_exist = inlist[5] 
	sgm_data = inlist[6]
	slidx = inlist[7]
	prngmin = inlist[8]
	prngmax = inlist[9]
	optalgo = inlist[10]
	rgalg = inlist[11]
	rg_trained = inlist[12]
	Ln = inlist[13]
	Lr = inlist[14]

	# Compute L-norm of tissue parameters contained in the grid search dictionary
	Ngrid = dict_pars.shape[0]
	ntissue = dict_pars.shape[1]
	dict_pars_Lnorm = np.zeros(Ngrid)
	for aa in range(0,Ngrid):
		norm_acc = 0.0
		for bb in range(0,ntissue):
			if(Ln==1):
				norm_acc = norm_acc + np.abs( 0.5 - (dict_pars[aa,bb] - prngmin[bb])/(prngmax[bb] - prngmin[bb]) )
			elif(Ln==2):
				norm_acc = norm_acc + ( 0.5 - (dict_pars[aa,bb] - prngmin[bb])/(prngmax[bb] - prngmin[bb]) )*( 0.5 - (dict_pars[aa,bb] - prngmin[bb])/(prngmax[bb] - prngmin[bb]) )
			else:
				raise RuntimeError('ERROR. An L{}-norm is not supported for objective function regularisation'.format(Ln))
		dict_pars_Lnorm[aa] = norm_acc/float(ntissue)

		
	# Allocate output maps
	Nii = m_data.shape[0]
	Njj = m_data.shape[1]
	pars_slice = np.zeros((Nii,Njj,ntissue))
	Exitmap_slice = np.zeros((Nii,Njj))      # Exit code will be 0 in the background, 1 where fitting was successful, -1 where fitting failed
	Fobj_slice = np.zeros((Nii,Njj))
	LogL_slice = np.zeros((Nii,Njj))
	BIC_slice = np.zeros((Nii,Njj))
	AIC_slice = np.zeros((Nii,Njj))

	# Loop through voxels
	for ii in range(0,Nii):
		for jj in range(0,Njj):
				
			## Process voxels within binary mask (if provided)
			if(k_data[ii,jj]==1):
				
				# Initialise exit code to 1 (success)
				Exflag = 1
				
				# Get actual MRI measurements and normalise them
				mvox = m_data[ii,jj,:]                # MRI measurements
						
				# Get noise level if it was provided
				if(sgm_exist):				
					sgm_vox = sgm_data[ii,jj]
				else:
					sgm_vox = 0.0

				# Prepare synthetic signals for dictionary fitting (grid search)
				Nmicro = dict_pars.shape[0]            # Number of combinations of tissue parameters in the grid search
				mvox_mat = np.tile(mvox,(Nmicro,1))    # Matrix of actual MRI measurements	

				# Find objective function values for grid search dictionary
				fobj_fidelity = (1.0 - Lr)*np.nanmean( (mvox_mat - np.sqrt(dict_sigs**2 + sgm_vox**2))**2 , axis=1) 
				fobj_regulariser = Lr*dict_pars_Lnorm
				fobj_array = fobj_fidelity + fobj_regulariser
						
				# Estimate tissue parameters via dictionary fitting
				try:
					# Get the position within the dictionary of the signal that most resembles our measurements
					min_idx = np.argmin(fobj_array)
					Fobj_grid = np.min(fobj_array)
						
					# Get microstructural parameters corresponding to the selected signal 
					param_init = dict_pars[min_idx,:]

					# Prepare parameter bounds and initial guess for function minimisation
					param_bound = []
					for yy in range(0,ntissue):
						param_bound.append((prngmin[yy],prngmax[yy]))

					# Perform model fitting
					modelfit = minimize(_fobj, param_init, method=optalgo, args=tuple([mvox,sgm_vox,navg,rgalg,rg_trained,Ln,Lr,prngmin,prngmax]), bounds=param_bound)
					param_fit = np.array(modelfit.x)
					Fobj = modelfit.fun
					if(Fobj>Fobj_grid):
							Exflag = -1

					# Compute log-likelihood, BIC and AIC if an estimate for the noise standard deviation was provided
					if(sgm_exist):
						Nmri = float(mvox.size)
						Nukn = float(ntissue)
						Fobj_noreg = _fobj(param_fit,mvox,sgm_vox,navg,rgalg,rg_trained,Ln,0.0,prngmin,prngmax)   # Value of obj. function with no regularisation for probabilistic calculations
						LogL = (-0.5/(sgm_vox*sgm_vox))*Fobj_noreg*Nmri - 0.5*Nmri*np.log( np.sqrt(2*np.pi*sgm_vox*sgm_vox) )   # Log-likelihood
						BIC = -2.0*LogL + Nukn*np.log(Nmri)      # Bayesian information criterion
						AIC = -2.0*LogL + 2.0*Nukn               # Akaike information criterion
					else:
						LogL = np.nan
						BIC = np.nan
						AIC = np.nan

					# Check if values make sense - check if any NaNs were obtained in the output parameter maps
					if( np.isnan(np.sum(param_fit))  ):
						param_fit = np.nan*np.zeros(ntissue)
						Fobj = np.nan
						LogL = np.nan
						BIC = np.nan
						AIC = np.nan
						Exflag = -1
							
				except:
					param_fit = np.nan*np.zeros(ntissue)
					Fobj = np.nan
					LogL = np.nan
					BIC = np.nan
					AIC = np.nan
					Exflag = -1
					

				# Store microstructural parameters for output
				for yy in range(0,ntissue):
					pars_slice[ii,jj,yy] = param_fit[yy]
				Exitmap_slice[ii,jj] = Exflag
				Fobj_slice[ii,jj] = Fobj
				LogL_slice[ii,jj] = LogL
				BIC_slice[ii,jj] = BIC
				AIC_slice[ii,jj] = AIC

	# Prepare output list and return it
	outlist = [pars_slice,Exitmap_slice,Fobj_slice,LogL_slice,BIC_slice,AIC_slice,slidx]
	return outlist
	


def run(mrifile, sdictfile, pdictfile, output, maskfile=None, noisefile=None, navg=1, Nword=0, pmin=None, pmax=None, regnorm=2, regw=0.001, nthread=1, slicedim=2, nlinalgo='trust-constr', rgalgo='rbf', rgpar=None):
	''' This tool performs maximum-likelihood fitting of a quantitative MRI signal model that is not known analytically, 
	    but that is approximated numerically given examples of signals and tissue parameters. 
	    
	    Third-party dependencies: nibabel, numpy, scipy. 
	    
	    Developed and validated with versions: nibabel 3.2.1, numpy 1.21.5, scipy 1.7.3. 
	    
	    Author: Francesco Grussu, Vall d Hebron Institute of Oncology (VHIO). 
	    Email: <fgrussu@vhio.net>
	    
	    USAGE
	    run(mrifile, sdictfile, pdictfile, output, maskfile=None, noisefile=None, navg=1, ...
		pmin=None, pmax=None, regnorm=2, regw=0.001, nthread=1, slicedim=2, ...
		nlinalgo='trust-constr', rgalgo='rbf', rgpar=None)
	    
	    * mrifile:    path of a 4D NIFTI file storing M quantitative MRI measurements acquired
	                  (e.g., diffusion MRI at different b-values and/or diffusion time; relaxometry at
	                  increasing echo times; etc). Each voxel should contain a signal defined in [0; 1].
	                  In diffusion MRI, this can be obtained by dividing each voxel for the signal at b = 0
	                  
	    * sdictfile:  path of a NumPy binary file (.npy) storing a dictionary of synthetic MRI measurements to be used for
	                  model fitting. The file should contain a variable storing a 2D numpy matrix of size Nmicro x M,
	                  where Nmicro is the number of example microstructures in the dictionary and Nmeas is the number of MRI 
	                  signal measurements.  This implies that different example microstructures are arranged along rows, 
	                  while the different MRI measurements coming from a given microstructure are arranged along columns. 
	                  NOTE 1: the quantitative MRI protocol used to generate "sdictfile" MUST match that used to acquire 
	                  the scan to fit stored in the "mrifile" NIFTI
	                  NOTE 2: the signal dictionary should be defined in [0; 1]
	    
	    * pdictfile:  path of a NumPy binary file (.npy) storing a dictionary of tissue parameters corresponding to the 
	                  signals stored in the signal dictionary "sdictfile", to be used for model fitting. 
	                  The file should contain a variable storing a 2D numpy matrix of size Nmicro x P,
	                  where Nmicro is the number of example microstructures in the dictionary and P is the number of 
	                  tissue parameters. This implies that different example microstructures are arranged along rows, while
	                  the values of the different tissue parameters of each microstructure are arranged along columns. 
	              
	    * output:     root file name of output files. Output NIFTIs will be stored as double-precision floating point images 
	                  (FLOAT64), and will store the estimated parametric maps. The number of parametric maps outputted depends
	                  on number of parameters contained in the tissue parameter dictionary "pdictfile". If that dictionary
	                  contains P tissue parameters, then there will be P output parameteric maps (one per each tissue parameter 
	                  of "pdictfile", in the same order). These will be stored as *_par1.nii, *_par2.nii, ...,  *_parP.nii.
					  
	                  
	                  Additionally, an output exit code map will also be stored as *_exit.nii (voxels containing -1: warning, 
	                  failure in non-linear fitting; voxels containing 0: background; voxels containing 1: 
	                  successful parameter estimation). 
	                  
	                  If a noise map was provided with the noisefile input parameter, additional output NIFTI files 
	                  storing quality of fit metrics are stored, i.e.: *_logL.nii (log-likelihood), 
	                  *_BIC.nii (Bayesian Information Criterion), and *_AIC.nii (Akaike Information Criterion) 
	                  
	    * maskfile:   3D mask in NIFTI format (computation will be performed only in voxels where mask = 1)
	                  
	    * noisefile:  3D noise standard deviation map in NIFTI format. If provided, the signal level will be compared to the
	                  an estimate of the noise floor when comparing the objective function (offset-Gaussian model).
	                  
	    * navg:       number of signal averages used for acquisition, which is used to get a more accurate estimate of the
	                  noise floor given the estimate of the noise standard deviation (default: 1; ignored if noisefile = None)
	                  Note that in some vendors, this parameter is referred to as number of excitations (NEX)
	    
	    * Nword:      number of values to test for each tissue parameter in the grid search. If set to 0, the
	                  tissue parameter dictionary contained in "pdictfile" used to learn the foward model
	                  will also be used for the grid search. If Nword > 0, then a uniformly-sampled grid will be generated.
	                  Default: 0 (i.e., use the same dictionary to learn the forward model and for the grid search)  		  

	    * pmin:       list or array of P elements storing the lower bounds for tissue parameters. The length of pmin must
	                  match the number of tissue parameters contained in the tissue parameter dictionary "pdictfile".
	                  This parameter can be used i) to select a subset of the dictionary contained in "pdictfile", i.e., to
	                  reduce the ranges of estimation for each tissue parameter, or ii) to extend the range of 
	                  estimation beyond min/max values contained in the dictionary. If pmin is not set, then the lower bounds
	                  of estimation will be the same as the min values contained in "pdictfile". 
	    
	    * pmax:       list or array of P elements storing the upper bounds for tissue parameters. The length of pmax must
	                  match the number of tissue parameters contained in the tissue parameter dictionary "pdictfile".
	                  This parameter can be used i) to select a subset of the dictionary contained in "pdictfile", i.e., to
	                  reduce the ranges of estimation for each tissue parameter, or ii) to extend the range of 
	                  estimation beyond min/max values contained in the dictionary. If pmax is not set, then the upper bounds
	                  of estimation will be the same as the min values contained in "pdictfile". 

	    * regnorm:    type of L-norm for regularisation (1 for LASSO, 2 for Tikhonov; default: 2) 
		
	    * regw:       weight of the regulariser (ranging in [0.0,1.0]; default: 0.001; set it to 0.0 for 
	                  standard non-linear least square fitting with no regularisation)      
	    
	    * nthread:    number of threads to be used for computation (default: 1, single thread)
	    
	    * slicedim:   image dimension along which parallel computation will be exectued when nthread > 1
	                  (can be 0, 1, 2 -- default slicedim=2, implying parallel processing along 3rd 
	                  image dimension) 
	    
	    * nlinalgo:   algorithm to be used for constrained objective function minimisation in non-linear fitting 
	                  (relevant if nlinfit = True). Choose among: 
	                  "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", and "trust-constr". 
	                  Default: "trust-constr" - see documentation of scipy.optimize.minimize for 
	                  information on each optimisation method)
	    
	    * rgalgo:     string specifying the algorithm to be used to derive a continuous representation of the forward model 
	                  signal = f(tissue parameters), given the input MRI sequence. Available options: 
	                  "rbf" (radial basis function regressor based on thin plate splines); 
	                  "linearND" (piece-wise linear interpolation in N dimensions). 
	                  Default: "rbf".
	    
	    * rgpar:      list or arry of hyperparameters of the regressor used to derive a continuous signal representation 
	                  for the forward signal model signal = f(tissue parameters) for the given MRI sequence. 
	                  A different number of hyperparameters may be needed for each regressor type, as detailed here: 
	                  rgalgo = "rbf" -->       hyperparameters are the smoothing factor and the degree of the added polynomial.
	                                           Default: rgpar = [1.0,1] 
	                  regalgo = "linearND" --> there is one hyperparemter that can be 0 or 1 (or False/True), 
	                                           indicating whethere the regressor should be defined on normalised inputs 
	                                           (if 1 or True) or not (if 0 or False).
	                                           Default: rgpar = False (anything different from 0 or False will be treated 
	                                           as 1 or True).
	    
	    Third-party dependencies: nibabel, numpy, scipy.
	    
	    Developed and validated with versions: 
	    - nibabel 3.2.1
	    - numpy 1.21.5
	    - scipy 1.7.3

	    Author: Francesco Grussu, Vall d'Hebron Institute of Oncology, November 2022
		    <fgrussu@vhio.net>'''


	### Get time
	timeinitial = time.time()
	
	### Get number of threads
	nthread = int(nthread)
	ncpu = mp.cpu_count()
	if( (ncpu - 1) < nthread):
		nthread = ncpu - 1

	if(nthread<0):
		nthread = ncpu - 1
		print('WARNING: negative number of threads -- using {} instead'.format(nthread))

	## Get slice dimension for parallel processing
	if( not( (slicedim==0) or (slicedim==1) or (slicedim==2) ) ):
		slicedim = 2
		print('WARNING: invalid image dimension for parallel processing -- using dimension={} ({}-th dimension)'.format(slicedim,slicedim+1))		
	
	### Load MRI measurements and check for consistency
	print('')
	print('    ... loading MRI measurements')
	try:
		m_obj = nib.load(mrifile)
	except:
		print('')
		raise RuntimeError('ERROR: the 4D input file {} does not exist or is not in NIFTI format.'.format(mrifile))
	m_data = m_obj.get_fdata()
	m_size = m_data.shape
	m_size = np.array(m_size)
	m_header = m_obj.header
	m_affine = m_header.get_best_affine()
	m_dims = m_obj.shape
	if m_size.size!=4:
		print('')
		raise RuntimeError('ERROR: the 4D input file {} is not a 4D NIFTI.'.format(mrifile))				 

	### Load mask and check for consistency
	if (maskfile is not None):
		print('')
		print('    ... loading mask')
		try:
			k_obj = nib.load(maskfile)
		except:
			print('')
			raise RuntimeError('ERROR: the 3D input file {} does not exist or is not in NIFTI format.'.format(maskfile))
		k_data = k_obj.get_fdata()
		k_size = k_data.shape
		k_size = np.array(k_size)
		k_header = k_obj.header
		k_affine = k_header.get_best_affine()
		k_dims = k_obj.shape
		if k_size.size!=3:
			print('')
			raise RuntimeError('ERROR: the 3D input file {} is not a 3D NIFTI.'.format(maskfile))
		if ( (np.sum(k_affine==m_affine)!=16) or (k_dims[0]!=m_dims[0]) or (k_dims[1]!=m_dims[1]) or (k_dims[2]!=m_dims[2]) ):
			print('')
			raise RuntimeError('ERROR: the header geometry of {} and {} do not match.'.format(mrifile,maskfile))
		k_data[k_data>0] = 1
		k_data[k_data<=0] = 0	
	else:
		k_data = np.ones((m_size[0],m_size[1],m_size[2]))
	
	### Load noise standard deviation map
	if (noisefile is not None):
		print('')
		print('    ... loading noise map')
		try:
			sgm_obj = nib.load(noisefile)
		except:
			print('')
			raise RuntimeError('ERROR: the 3D input file {} does not exist or is not in NIFTI format.'.format(noisefile))
		sgm_data = sgm_obj.get_fdata()
		sgm_size = sgm_data.shape
		sgm_size = np.array(sgm_size)
		sgm_header = sgm_obj.header
		sgm_affine = sgm_header.get_best_affine()
		sgm_dims = sgm_obj.shape
		if sgm_size.size!=3:
			print('')
			raise RuntimeError('ERROR: the 3D input file {} is not a 3D NIFTI.'.format(noisefile))
		if ( (np.sum(sgm_affine==m_affine)!=16) or (sgm_dims[0]!=m_dims[0]) or (sgm_dims[1]!=m_dims[1]) or (sgm_dims[2]!=m_dims[2]) ):
			print('')
			raise RuntimeError('ERROR: the header geometry of {} and {} do not match.'.format(mrifile,noisefile))
		
		# Flags
		sgm_exist = True
		sgm_exist = True
			
	else:
		# Flags
		sgm_exist = False
		sgm_exist = False
		
		# Empty variable
		sgm_data = None
	
	
	## Load signal and tissue parameter dictionaries, and check for consistency
	try:
		dict_sigs = np.load(sdictfile)
	except:
		raise RuntimeError('ERROR: invalid signal dictionary file {}'.format(sdictfile))
			
	try:
		dict_pars = np.load(pdictfile)
	except:
		raise RuntimeError('ERROR: invalid tissue parameter dictionary file {}'.format(pdictfile))

	if(dict_sigs.shape[1]!=m_size[3]):
		raise RuntimeError('ERROR: the number of signal measurements in the signal dictionary {} does not match that in the NIFTI file {}'.format(sdictfile,mrifile))

	if(dict_sigs.shape[0]!=dict_pars.shape[0]):
		raise RuntimeError('ERROR: the numbers of unique microstructures in the signal and tissue parameter dictionaries ({} and {}) do not match'.format(sdictfile,pdictfile))
	
	if(np.max(dict_sigs)>1.0):
		raise RuntimeError('ERROR: the maximum signal magnitude in the signal dictionary {} should not be > 1.0'.format(sdictfile))
	if(np.min(dict_sigs)<0.0):
		raise RuntimeError('ERROR: the minimum signal magnitude in the signal dictionary {} should not be < 0.0'.format(sdictfile))


		
	## Manage tissue parameter lower and upper bounds
	npars = dict_pars.shape[1]
	if pmin is not None:
		pmin = np.array(pmin)
		pmin_size = pmin.size
		if(pmin_size!=npars):
			raise RuntimeError('ERROR: the number of tissue parameters lower bounds does not match the number of tissue parameters in the dictionary {}'.format(pdictfile))
	
	if pmax is not None:
		pmax = np.array(pmax)
		pmax_size = pmax.size
		if(pmax_size!=npars):
			raise RuntimeError('ERROR: the number of tissue parameters upper bounds does not match the number of tissue parameters in the dictionary {}'.format(pdictfile))
	
	# Remove microstructures from both signal and tissue parameter dictionaries that are not included in the desired tissue parameter range
	if pmin is None:
		pmin = np.zeros(npars)
		for qq in range(0,npars):
			pmin[qq] = np.min(dict_pars[:,qq])
	
	if pmax is None:
		pmax = np.zeros(npars)
		for qq in range(0,npars):
			pmax[qq] = np.max(dict_pars[:,qq])
		
	keeplist_all = np.ones(dict_pars.shape[0],dtype='bool')
	for qq in range(0,npars):	                             # This will not change the dictionary if no pmin or pmax bounds were provided
		parray = dict_pars[:,qq]
		keeplist = ~( (parray<pmin[qq]) | (parray>pmax[qq]) ) # True where tissue parameters fall within the requested range, False otherwise
		keeplist_all = keeplist & keeplist_all                # All conditions on all tissue parameters must be satisfied - this is a logical AND	 
	final_dict_sigs = dict_sigs[keeplist_all,:]               # Final dictionary of signals
	final_dict_pars = dict_pars[keeplist_all,:]               # Final dictionary of tissue parameters

	## Legacy variable name change
	final_dict_sigs_rescaled = np.copy(final_dict_sigs)

	## Train a regressor using the paired examples tissue parameters - signals from the discrete dictionary (but only if non-linear fitting beyond grid search is required)
	print('')
	print('    ... training a {} regressor to approximate the forward model signal = f(tissue parameters)'.format(rgalgo))
	if(rgalgo=='rbf'):
		if rgpar is None:
			rgpar = np.array([1.0,1])       # Smoothing factor and degree of the added polynomial (default is smoothing = 1.0, degree = 1)
		reg_trained = _train_rbf( final_dict_pars, final_dict_sigs_rescaled, float(rgpar[0]), int(rgpar[1]) )
	elif(rgalgo=='linearND'):
		if rgpar is None:
			rgpar = False                   # Build the interpolator on normalised coordinates (if True) or not (if False; default is False)
		reg_trained = _train_linearND( final_dict_pars, final_dict_sigs_rescaled, bool( np.array([rgpar]) ) )
	else:
		raise RuntimeError('ERROR: unkown regression algorithm {}'.format(rgalgo))

	## Generate uniformly-spaced dictionaries for the grid search, as one would do for standard fitting of analytical models
	
	# Get tissue parameter grid
	gridres = int(Nword)
	
	# If the grid depth is 0, the user has flagged that for the grid search they want to use the same dictionary employed to learn the forward model p --> S(p)
	if(gridres==0):
		pdict_uniform = np.copy(final_dict_pars)
		sdict_uniform = np.copy(final_dict_sigs_rescaled)
	
	# If the grid depth is not 0, the user has flagged that for the grid search they want to sample uniformly the tissue parameter space
	else:
		if(npars==1):
			p1_array = np.linspace(pmin[0],pmax[0],num=gridres)
			pdict_uniform = np.zeros((gridres,1))
			pdict_uniform[0,:] = p1_array
		elif(npars==2):
			p1_array = np.linspace(pmin[0],pmax[0],num=gridres)
			p2_array = np.linspace(pmin[1],pmax[1],num=gridres)
			p1_mat, p2_mat = np.meshgrid(p1_array,p2_array)
			pdict_uniform = np.zeros((p1_mat.size,2))
			pdict_uniform[:,0] = p1_mat.flatten()
			pdict_uniform[:,1] = p2_mat.flatten()
		elif(npars==3):
			p1_array = np.linspace(pmin[0],pmax[0],num=gridres)
			p2_array = np.linspace(pmin[1],pmax[1],num=gridres)
			p3_array = np.linspace(pmin[2],pmax[2],num=gridres)
			p1_mat, p2_mat, p3_mat = np.meshgrid(p1_array,p2_array,p3_array)
			pdict_uniform = np.zeros((p1_mat.size,3))
			pdict_uniform[:,0] = p1_mat.flatten()
			pdict_uniform[:,1] = p2_mat.flatten()
			pdict_uniform[:,2] = p3_mat.flatten()
		elif(npars==4):
			p1_array = np.linspace(pmin[0],pmax[0],num=gridres)
			p2_array = np.linspace(pmin[1],pmax[1],num=gridres)
			p3_array = np.linspace(pmin[2],pmax[2],num=gridres)
			p4_array = np.linspace(pmin[3],pmax[3],num=gridres)
			p1_mat, p2_mat, p3_mat, p4_mat = np.meshgrid(p1_array,p2_array,p3_array,p4_array)
			pdict_uniform = np.zeros((p1_mat.size,4))
			pdict_uniform[:,0] = p1_mat.flatten()
			pdict_uniform[:,1] = p2_mat.flatten()
			pdict_uniform[:,2] = p3_mat.flatten()
			pdict_uniform[:,3] = p4_mat.flatten()
		elif(npars==5):
			p1_array = np.linspace(pmin[0],pmax[0],num=gridres)
			p2_array = np.linspace(pmin[1],pmax[1],num=gridres)
			p3_array = np.linspace(pmin[2],pmax[2],num=gridres)
			p4_array = np.linspace(pmin[3],pmax[3],num=gridres)
			p5_array = np.linspace(pmin[4],pmax[4],num=gridres)
			p1_mat, p2_mat, p3_mat, p4_mat, p5_mat = np.meshgrid(p1_array,p2_array,p3_array,p4_array,p5_array)
			pdict_uniform = np.zeros((p1_mat.size,5))
			pdict_uniform[:,0] = p1_mat.flatten()
			pdict_uniform[:,1] = p2_mat.flatten()
			pdict_uniform[:,2] = p3_mat.flatten()
			pdict_uniform[:,3] = p4_mat.flatten()
			pdict_uniform[:,4] = p5_mat.flatten()
		elif(npars==6):
			p1_array = np.linspace(pmin[0],pmax[0],num=gridres)
			p2_array = np.linspace(pmin[1],pmax[1],num=gridres)
			p3_array = np.linspace(pmin[2],pmax[2],num=gridres)
			p4_array = np.linspace(pmin[3],pmax[3],num=gridres)
			p5_array = np.linspace(pmin[4],pmax[4],num=gridres)
			p6_array = np.linspace(pmin[5],pmax[5],num=gridres)
			p1_mat, p2_mat, p3_mat, p4_mat, p5_mat, p6_mat = np.meshgrid(p1_array,p2_array,p3_array,p4_array,p5_array,p6_array)
			pdict_uniform = np.zeros((p1_mat.size,6))
			pdict_uniform[:,0] = p1_mat.flatten()
			pdict_uniform[:,1] = p2_mat.flatten()
			pdict_uniform[:,2] = p3_mat.flatten()
			pdict_uniform[:,3] = p4_mat.flatten()
			pdict_uniform[:,4] = p5_mat.flatten()
			pdict_uniform[:,5] = p6_mat.flatten()
		elif(npars==7):
			p1_array = np.linspace(pmin[0],pmax[0],num=gridres)
			p2_array = np.linspace(pmin[1],pmax[1],num=gridres)
			p3_array = np.linspace(pmin[2],pmax[2],num=gridres)
			p4_array = np.linspace(pmin[3],pmax[3],num=gridres)
			p5_array = np.linspace(pmin[4],pmax[4],num=gridres)
			p6_array = np.linspace(pmin[5],pmax[5],num=gridres)
			p7_array = np.linspace(pmin[6],pmax[6],num=gridres)
			p1_mat, p2_mat, p3_mat, p4_mat, p5_mat, p6_mat, p7_mat = np.meshgrid(p1_array,p2_array,p3_array,p4_array,p5_array,p6_array,p7_array)
			pdict_uniform = np.zeros((p1_mat.size,7))
			pdict_uniform[:,0] = p1_mat.flatten()
			pdict_uniform[:,1] = p2_mat.flatten()
			pdict_uniform[:,2] = p3_mat.flatten()
			pdict_uniform[:,3] = p4_mat.flatten()
			pdict_uniform[:,4] = p5_mat.flatten()
			pdict_uniform[:,5] = p6_mat.flatten()
			pdict_uniform[:,6] = p7_mat.flatten()
		elif(npars==8):
			p1_array = np.linspace(pmin[0],pmax[0],num=gridres)
			p2_array = np.linspace(pmin[1],pmax[1],num=gridres)
			p3_array = np.linspace(pmin[2],pmax[2],num=gridres)
			p4_array = np.linspace(pmin[3],pmax[3],num=gridres)
			p5_array = np.linspace(pmin[4],pmax[4],num=gridres)
			p6_array = np.linspace(pmin[5],pmax[5],num=gridres)
			p7_array = np.linspace(pmin[6],pmax[6],num=gridres)
			p8_array = np.linspace(pmin[7],pmax[7],num=gridres)
			p1_mat, p2_mat, p3_mat, p4_mat, p5_mat, p6_mat, p7_mat, p8_mat = np.meshgrid(p1_array,p2_array,p3_array,p4_array,p5_array,p6_array,p7_array,p8_array)
			pdict_uniform = np.zeros((p1_mat.size,8))
			pdict_uniform[:,0] = p1_mat.flatten()
			pdict_uniform[:,1] = p2_mat.flatten()
			pdict_uniform[:,2] = p3_mat.flatten()
			pdict_uniform[:,3] = p4_mat.flatten()
			pdict_uniform[:,4] = p5_mat.flatten()
			pdict_uniform[:,5] = p6_mat.flatten()
			pdict_uniform[:,6] = p7_mat.flatten()
			pdict_uniform[:,7] = p8_mat.flatten()
		else:
			raise RuntimeError('ERROR. Currently this tool supports the estimation of a maximum of 8 tissue parameters from MRI measurements.')

		# Get signal grid corresponding to uniformly-sampled parameters	
		sdict_uniform = np.zeros((pdict_uniform.shape[0],final_dict_sigs_rescaled.shape[1]))
		for gg in range(0,pdict_uniform.shape[0]):
			sdict_uniform[gg,:] = _sig(pdict_uniform[gg,:],rgalgo,reg_trained)

	### Allocate output parametric maps
	Tparmap = np.zeros((m_size[0],m_size[1],m_size[2],npars))     # Allocate output: parametric maps to be estimated
	Exitmap = np.zeros((m_size[0],m_size[1],m_size[2]))           # Allocate output:  exit code map
	Fobjmap = np.zeros((m_size[0],m_size[1],m_size[2]))           # Allolcate output: Fobj map
	LogLmap = np.zeros((m_size[0],m_size[1],m_size[2]))           # Allolcate output: log-likelihood map
	BICmap = np.zeros((m_size[0],m_size[1],m_size[2]))            # Allolcate output: Bayesian Information Criterion map
	AICmap = np.zeros((m_size[0],m_size[1],m_size[2]))            # Allolcate output: Akaike Information Criterion map
	
	### Processing
	print('')
	print('    ... processing -- please wait')
	
	# Prepare information for current MRI slice
	inputlist = [] 
	for ww in range(0,m_size[slicedim]):

		if(slicedim==0):
		
			k_data_sl = k_data[ww,:,:]
			m_data_sl = m_data[ww,:,:,:]
			if(sgm_exist):
				sgm_data_sl = sgm_data[ww,:,:]
			else:
				sgm_data_sl = None
		
		elif(slicedim==1):
		
			k_data_sl = k_data[:,ww,:]
			m_data_sl = m_data[:,ww,:,:]
			if(sgm_exist):
				sgm_data_sl = sgm_data[:,ww,:]
			else:
				sgm_data_sl = None
		
		elif(slicedim==2):
		
			k_data_sl = k_data[:,:,ww]
			m_data_sl = m_data[:,:,ww,:]
			if(sgm_exist):
				sgm_data_sl = sgm_data[:,:,ww]	
			else:
				sgm_data_sl = None
			
		else:
			raise RuntimeError('ERROR: invalid slice dimension slicedim = {}'.format(slicedim)) 

		
		#sliceinfo = [navg,final_dict_pars,final_dict_sigs_rescaled,k_data_sl,m_data_sl,sgm_exist,sgm_data_sl,ww,pmin,pmax,nlinalgo,rgalgo,reg_trained,regnorm,regw]
		sliceinfo = [navg,pdict_uniform,sdict_uniform,k_data_sl,m_data_sl,sgm_exist,sgm_data_sl,ww,pmin,pmax,nlinalgo,rgalgo,reg_trained,regnorm,regw]
		inputlist.append(sliceinfo)

	
	# Send slice to process in parallel if nthread>1	
	if(nthread>1):
		
		# Create the parallel pool and give jobs to the workers
		fitpool = mp.Pool(processes=nthread)  # Create parallel processes
		fitpool_pids_initial = [proc.pid for proc in fitpool._pool]  # Get PIDs
		fitresults = fitpool.map_async(_procslice,inputlist)          # Send off to parallel processing
			
		# Check whether results are ready
		while not fitresults.ready():
			fitpool_pids_new = [proc.pid for proc in fitpool._pool]  # Get PIDs again
			if fitpool_pids_new!=fitpool_pids_initial:
				print('')
				raise RuntimeError('ERROR: some processes died during parallel fitting.') # Error: some processes where killed, stop everything and throw an error 
			
		# Work done: get results
		fitlist = fitresults.get()

		# Collect fitting output and re-assemble MRI slices	## Recall: the output of _procslice() contains out = [pars_slice,Exitmap_slice,Fobj_slice,LogL_slice,BIC_slice,AIC_slice,slidx]	
		for ww in range(0, m_size[slicedim]):
			myresults = fitlist[ww]		
			allmaps_sl = myresults[0]	
			Exitmap_sl = myresults[1]
			Fobj_sl = myresults[2]
			LogL_sl = myresults[3]
			BIC_sl = myresults[4]
			AIC_sl = myresults[5]	
			myslice = myresults[6]		

			if(slicedim==0):
				for uu in range(0,npars):
					Tparmap[myslice,:,:,uu] = allmaps_sl[:,:,uu]
				Exitmap[myslice,:,:] = Exitmap_sl
				Fobjmap[myslice,:,:] = Fobj_sl
				LogLmap[myslice,:,:] = LogL_sl
				BICmap[myslice,:,:] = BIC_sl
				AICmap[myslice,:,:] = AIC_sl
				
			elif(slicedim==1):
				for uu in range(0,npars):
					Tparmap[:,myslice,:,uu] = allmaps_sl[:,:,uu]
				Exitmap[:,myslice,:] = Exitmap_sl
				Fobjmap[:,myslice,:] = Fobj_sl
				LogLmap[:,myslice,:] = LogL_sl
				BICmap[:,myslice,:] = BIC_sl
				AICmap[:,myslice,:] = AIC_sl			
		
			elif(slicedim==2):
				for uu in range(0,npars):
					Tparmap[:,:,myslice,uu] = allmaps_sl[:,:,uu]
				Exitmap[:,:,myslice] = Exitmap_sl
				Fobjmap[:,:,myslice] = Fobj_sl
				LogLmap[:,:,myslice] = LogL_sl
				BICmap[:,:,myslice] = BIC_sl
				AICmap[:,:,myslice] = AIC_sl
							
			else:
				raise RuntimeError('ERROR: invalid slice dimension slicedim = {}'.format(slicedim)) 		
			
		
	
	# Single CPU at work	
	else:
	
		for ww in range(0, m_size[slicedim]):
			myresults = _procslice(inputlist[ww]) 
			allmaps_sl = myresults[0]	
			Exitmap_sl = myresults[1]
			Fobj_sl = myresults[2]
			LogL_sl = myresults[3]
			BIC_sl = myresults[4]
			AIC_sl = myresults[5]	
			myslice = myresults[6]	
			
			if(slicedim==0):
				for uu in range(0,npars):
					Tparmap[myslice,:,:,uu] = allmaps_sl[:,:,uu]
				Exitmap[myslice,:,:] = Exitmap_sl
				Fobjmap[myslice,:,:] = Fobj_sl
				LogLmap[myslice,:,:] = LogL_sl
				BICmap[myslice,:,:] = BIC_sl
				AICmap[myslice,:,:] = AIC_sl
				
			elif(slicedim==1):
				for uu in range(0,npars):
					Tparmap[:,myslice,:,uu] = allmaps_sl[:,:,uu]
				Exitmap[:,myslice,:] = Exitmap_sl
				Fobjmap[:,myslice,:] = Fobj_sl
				LogLmap[:,myslice,:] = LogL_sl
				BICmap[:,myslice,:] = BIC_sl
				AICmap[:,myslice,:] = AIC_sl			
		
			elif(slicedim==2):
				for uu in range(0,npars):
					Tparmap[:,:,myslice,uu] = allmaps_sl[:,:,uu]
				Exitmap[:,:,myslice] = Exitmap_sl
				Fobjmap[:,:,myslice] = Fobj_sl
				LogLmap[:,:,myslice] = LogL_sl
				BICmap[:,:,myslice] = BIC_sl
				AICmap[:,:,myslice] = AIC_sl
							
			else:
				raise RuntimeError('ERROR: invalid slice dimension slicedim = {}'.format(slicedim)) 	
		
	
			
	### Save output NIFTIs				
	print('')
	print('    ... saving output files')
	buffer_header = m_obj.header
	buffer_header.set_data_dtype('float64')   # Make sure we save output files float64, even if input is not
	
	# Save exit code and objective function
	exit_obj = nib.Nifti1Image(Exitmap,m_obj.affine,buffer_header)
	nib.save(exit_obj, '{}_exit.nii'.format(output))
	
	fobj_obj = nib.Nifti1Image(Fobjmap,m_obj.affine,buffer_header)
	nib.save(fobj_obj, '{}_fobj.nii'.format(output))
		
	# Save parametric maps
	for uu in range(0,npars):
		niiout_obj = nib.Nifti1Image(np.squeeze(Tparmap[:,:,:,uu]),m_obj.affine,buffer_header)
		nib.save(niiout_obj, '{}_par{}.nii'.format(output,uu+1))
	
	# If a noise map had been provided, we can get LogL, BIC and AIC from FObj and store them
	if(sgm_exist):
		logL_obj = nib.Nifti1Image(LogLmap,m_obj.affine,buffer_header)
		nib.save(logL_obj, '{}_logL.nii'.format(output))
		
		bic_obj = nib.Nifti1Image(BICmap,m_obj.affine,buffer_header)
		nib.save(bic_obj, '{}_BIC.nii'.format(output))
		
		aic_obj = nib.Nifti1Image(AICmap,m_obj.affine,buffer_header)
		nib.save(aic_obj, '{}_AIC.nii'.format(output))

	### Done
	timefinal = time.time()
	print('    ... done - it took {} sec'.format(timefinal - timeinitial))
	print('')




# Run the module as a script when required
if __name__ == "__main__":

	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='This tool performs maximum-likelihood fitting of a quantitative MRI signal model that is not known analytically, but that is approximated numerically given examples of signals and tissue parameters. Third-party dependencies: nibabel, numpy, scipy. Developed and validated with versions: nibabel 3.2.1, numpy 1.21.5, scipy 1.7.3. Author: Francesco Grussu, Vall d Hebron Institute of Oncology (VHIO). Email: <fgrussu@vhio.net>.')
	parser.add_argument('s_file', help='path of a 4D NIFTI file storing M quantitative MRI measurements acquired (e.g., diffusion MRI at different b-values and/or diffusion time; relaxometry at increasing echo times; etc). Each voxel should contain a signal defined in [0; 1]. In diffusion MRI, this can be obtained by dividing each voxel for the signal at b = 0')
	parser.add_argument('sig_dict', help='path of a NumPy binary file (.npy) storing a dictionary of synthetic MRI measurements to be used formodel fitting. The file should contain a variable storing a 2D numpy matrix of size Nmicro x M,where Nmicro is the number of example microstructures in the dictionary and Nmeas is the number of MRIsignal measurements.  This implies that different example microstructures are arranged along rows,while the different MRI measurements coming from a given microstructure are arranged along columns. NOTE 1: the quantitative MRI protocol used to generate "sdictfile" MUST match that used to acquirethe scan to fit stored in the "mrifile" NIFTI. NOTE 2: the signal dictionary should be defined in [0; 1].')
	parser.add_argument('par_dict', help='path of a NumPy binary file (.npy) storing a dictionary of tissue parameters corresponding to the signals stored in the signal dictionary "sdictfile", to be used for model fitting. The file should contain a variable storing a 2D numpy matrix of size Nmicro x P, where Nmicro is the number of example microstructures in the dictionary and P is the number of tissue parameters. This implies that different example microstructures are arranged along rows, while the values of the different tissue parameters of each microstructure are arranged along columns.')
	parser.add_argument('out', help='root file name of output files. Output NIFTIs will be stored as double-precision floating point images (FLOAT64), and will store the estimated parametric maps. The number of parametric maps outputted depends on number of parameters contained in the tissue parameter dictionary "pdictfile". If that dictionary contains P tissue parameters, then there will be P output parameteric maps (one per each tissue parameter of "pdictfile", in the same order). These will be stored as *_par1.nii, *_par2.nii, ...,  *_parP.nii Additionally, an output exit code map will also be stored as *_exit.nii (voxels containing -1: warning, failure in non-linear fitting; voxels containing 0: background; voxels containing 1: successful parameter estimation). If a noise map was provided with the noisefile input parameter, additional output NIFTI files storing quality of fit metrics are stored, i.e.: *_logL.nii (log-likelihood), *_BIC.nii (Bayesian Information Criterion), and *_AIC.nii (Akaike Information Criterion).')
	parser.add_argument('--mask', metavar='<file>', help='3D mask in NIFTI format (computation will be performed only in voxels where mask = 1)')
	parser.add_argument('--noise', metavar='<file>', help='3D noise standard deviation map in NIFTI format. If provided, the signal level will be compared to the an estimate of the noise floor when comparing the objective function (offset-Gaussian model).')
	parser.add_argument('--savg', metavar='<num>', default='1', help='number of signal averages used for MRI data acquisition (default: 1). This parameter is used for the estimation of the noise floor (it is ignored if the option --noise is not used). Note that in some vendors, this parameter is also referred to as number of excitations (NEX).')
	parser.add_argument('--nw', metavar='<num>', default='0', help='number of values to test for each unknown tissue parameter in the grid search (it must be an integer). If set to 0, the tissue parameter dictionary contained in "par_dict" used to learn the foward model will be used also for the grid search. If > 0, then a uniformly-sampled grid will be generated. Default: 0 (i.e., use the same dictionary to learn the forward model and for the grid search)')
	parser.add_argument('--pmin', metavar='<list>', help='comma-separated list of P elements storing the lower bounds for tissue parameters. The length of the list must match the number of tissue parameters contained in the tissue parameter dictionary "par_dict".This option can be used i) to select a subset of the dictionary contained in "par_dict", i.e., to reduce the ranges for estimation for each tissue parameters, or ii) to extend the range of estimation beyond the min/max values contained in the dictionary. If not set, then the lower bounds contained in the dictionary "par_dict" will be used.')
	parser.add_argument('--pmax', metavar='<list>', help='comma-separated list of P elements storing the upper bounds for tissue parameters. The length of the list must match the number of tissue parameters contained in the tissue parameter dictionary "par_dict".This option can be used i) to select a subset of the dictionary contained in "par_dict", i.e., to reduce the ranges for estimation for each tissue parameters, or ii) to extend the range of estimation beyond the min/max values contained in the dictionary. If not set, then the upper bounds contained in the dictionary "par_dict" will be used.')
	parser.add_argument('--reg', metavar='<Lnorm,weight>', help='comma-separated list of parameters for fitting regularisation specifying i) the type of L-norm (1 for LASSO, 2 for Tikhonov), ii) the weight of the regulariser, ranging in [0.0,1.0]. Default: 2,0.001 (L2 norm, with a weight of 0.001). Set 2,0.0 for a standard non-linear least square fitting with no regularisation.')
	parser.add_argument('--ncpu', metavar='<num>', default='1', help='number of threads to be used for computation (default: 1, single thread)')
	parser.add_argument('--sldim', metavar='<num>', default='2', help='image dimension along which parallel computation will be exectued when nthread > 1 (it can be 0, 1, 2; default 2, implying parallel processing along the 3rd image dimension)')
	parser.add_argument('--nlalgo', metavar='<string>', default='trust-constr', help='algorithm to be used for constrained objective function minimisation in non-linear fitting (relevant if --nlfit 1). Choose among: "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", and "trust-constr" (default: "trust-constr" - see documentation of scipy.optimize.minimize for information on the optimisation algorithm)')
	parser.add_argument('--rgalgo', metavar='<string>', default='rbf', help='Algorithm to be used to derive a continuous representation of the forward model signal = f(tissue parameters), given the input MRI sequence. Available options: "rbf" (radial basis function regressor based on thin plate splines); "linearND" (piece-wise linear interpolation in N dimensions). Default: "rbf".')
	parser.add_argument('--rgpars', metavar='<list>', help='list of comma-separated hyperparameters of the regressor used to derive a continuous signal representation for the forward signal model signal = f(tissue parameters) for the given MRI sequence. A different number of hyperparameters may be needed for each regressor type, as detailed here: for --regalgo "rbf", hyperparameters are the smoothing factor and the degree of the added polynomial (default: --regpars 1.0,1); for --regalgo "linearND", there is one hyperparemter that can be 0 or 1, indicating whethere the regressor should be defined on normalised inputs (if 1) or not (if 0) (default: --regpars 0; anything different from 0 will be treated as 1).')
	args = parser.parse_args()


	### Get input arguments
	sfile = args.s_file
	sigdict = args.sig_dict
	pardict = args.par_dict
	outroot = args.out
	maskfile = args.mask
	sgmfile = args.noise
	nex = int(args.savg)
	nword = int(args.nw)
	cpucnt = int(args.ncpu)
	slptr = int(args.sldim)
	nlopt = args.nlalgo
	regpars = args.rgpars
	regalgo = args.rgalgo

	# Regularisation options
	if args.reg is not None:
		fooreg = (args.reg).split(',')
		fooreg = np.array( list(map( float, fooreg )) )
		Lnorm = int(fooreg[0])
		Lweight = float(fooreg[1])
	else:
		Lnorm = 2
		Lweight = 0.001

	# Lower bounds
	if args.pmin is not None:
		pMin = (args.pmin).split(',')
		pMin = np.array( list(map( float, pMin )) )
	else:
		pMin = None
	
	# Upper bounds
	if args.pmax is not None:
		pMax = (args.pmax).split(',')
		pMax = np.array( list(map( float, pMax )) )
	else:
		pMax = None
	
	
	# Non-linear fitting options and regressor properties
	if regpars is not None:
		if(regalgo=='rbf'):
			regpars = (args.regpars).split(',')
			regpars = np.array( list(map( float, regpars )) )
		elif(regalgo=='linearND'):
			regpars = int(regpars)
		else:
			raise RuntimeError('ERROR: unkown regression algorithm {}'.format(regalgo))	
	else:
		if(regalgo=='rbf'):
			regpars = np.array([1.0,1])  # Smoothing factor and degree of the added polynomial (default: smoothing=1.0, degree=1)
		elif(regalgo=='linearND'):
			regpars = int(0)             # Normalise inputs for ND interpolation or not (default: normalise=0, i.e. do not normalise)
		else:
			raise RuntimeError('ERROR: unkown regression algorithm {}'.format(regalgo))

	
	
	### Print feedback
	print('')
	print('***********************************************************************')
	print('                           mri2micro_dictml.py                         ')
	print('***********************************************************************')
	print('')
	print('** 4D NIFTI file with MRI measurements: {}'.format(sfile))
	print('** MRI signal dictionary: {}'.format(sigdict))
	print('** Tissue parameter dictionary: {}'.format(pardict))
	print('** Algorithm for signal = f(tissue parameters) regression (forward model): {}'.format(regalgo))
	print('** Forward regressor {} hyperparameters: {}'.format(regalgo,regpars))
	print('** Non-linear fitting constrained minimisation algorithm (inverse model): {}'.format(nlopt))
	print('** Number of threads for parallel slice processing: {}'.format(cpucnt))
	print('** Slice dimension for parallel processing: {}'.format(slptr))
	print('** Number of words for each tissue parameter grid search: {}'.format(nword))
	print('** Lower bound for tissue parameters: {}'.format(pMin))
	print('** Upper bound for tissue parameters: {}'.format(pMax))
	print('** Fitting regularisation options: Lnorm = {}, weight = {}'.format(Lnorm,Lweight))
	if( maskfile is not None ):
		print('** Optional binary mask file: {}'.format(maskfile))
	if( sgmfile is not None ):
		print('** Optional 3D NIFTI file with noise standard deviation map: {}'.format(sgmfile))
		print('** Number of signal averages: {}'.format(nex))
	print('** Output root name: {}'.format(outroot))
		

	### Run computation
	run(sfile, sigdict, pardict, outroot, maskfile=maskfile, noisefile=sgmfile, navg=nex, pmin=pMin, pmax=pMax, regnorm=Lnorm, regw=Lweight, nthread=cpucnt, slicedim=slptr, nlinalgo=nlopt, rgalgo=regalgo, rgpar=regpars)

	### Job done
	sys.exit(0)

