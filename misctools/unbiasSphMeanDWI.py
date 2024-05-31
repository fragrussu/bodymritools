### Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain). 
#   All rights reserved.
#   This code is distributed under the Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)).
#   Its use MUST also comply with the requirements of the individual licenses of all its dependencies.

import nibabel as nib
import numpy as np
import sys, argparse


def unbiasSM(dwi,noise,outfile,nsa=1,bv=None):
    '''Mitigate the noise floor bias in spherical mean diffusion-weighted signals.
       
       USAGE:
       unbiasSM(dwi,noise,outfile,nsa=1,bv=None)
       
       INPUT PARAMETERS

       * dwi      : path of a 4D NIFTI file storing diffusion MRI data. 
                    The NIFTI file can either 
                    - already contain spherical mean data 
                      at fixed b-value, or 
                    --  contain images acquired at different
                        gradient directions; in this case, the spherical mean will be 
                        computed. 
                    The default is -; to flag that the input images corresponds to 
                    different gradient directions, pass a b-value
                    list via optional input parameter bv (see below).

       * noise    :  path of a 3D NIFTI file storing a voxel-wise noise map
                     (standard deviation of Gaussian/Rician noise). This can be obained, 
                     for example, via Marchenko-Pastur Principal Component 
                     Analysis, as in Veraart J et al, NeuroImage 2016.
       
       * outfile  :  path of a 4D NIFTI file storing the unbiased spherical mean. 
                     If the input file contained images corresponding to different 
                     gradient directions, as flagged by passing a b-value file with
                     input bv, then the output file will have fewer volumes
                     than the inut file, as it will contain spherical mean signals. 
                     In this case, additional text file, where "*.bval_sph_mean" is appended 
                     to the output file name, will be saved, to store the b-values 
                     of the spherical mean signals.
       
       * nsa     :   total number of signal averages performed on the image on which 
                     the noise map was calculated. For example, if 1 b-value is acquired with 
                     NEX=2 using trace imaging (acquisition of 3 orthogonal gradients, 
                     each with NEX=2, which are then averaged by the scanner), then NSA = 6 
                     (2 averages per gradients, with averaging of gradients 
                     performed by the scanner). Default: nsa=1.
       
       * bv       :  path a a text file storing b-values (FSL format; in s/mm2; one raw, 
                     space-separated). If provided, the code will assume that the input NIFTI
                     file contains images corresponding to different gradient directions, so 
                     that the spherical mean needs calculating, before unbiasing can be 
                     performed. Spherical mean signals will be calcualated by averaging images
                     with the same b-value. Default: None, i.e., no b-value list provided, implying that 
                     the input NIFTI file stores spherical mean signals.

       
       This tools mitigated the noise floor bias using the analytical expression of 
       Pieciak T. et al, Proc ISMRM 2022, p.3900 (SM_unbiased = SM - 0.5*sigma^2/SM). 
       
       Author: Francesco Grussu, July 2023. 
       Email: <fgrussu@vhio.net>. 
       
       Third-party dependencies: numpy, nibabel. 
       Developed with numpy 1.23.5 and nibabel 5.1.0.
    
    '''

    ### Load diffusion data
    print('')
    print('    ... loading diffusion MRI measurements')
    try:
        m_obj = nib.load(dwi)
    except:
        print('')
        raise RuntimeError('ERROR: the input diffusion MRI file {} does not exist or is not in NIFTI format.'.format(dwi))
    m_data = m_obj.get_fdata()
    m_size = m_data.shape
    m_size = np.array(m_size)
    m_header = m_obj.header
    m_affine = m_header.get_best_affine()
    m_dims = m_obj.shape
    if m_size.size==3:
        print('WARNING: your diffusion MRI data set has only one volume')
    elif m_size.size==4:
        pass
    else:
        raise RuntimeError('ERROR: the diffusion MRI file {} is 2D or has dimensions > 4D. I do not know what to do.'.format(dwi))


    ### Load noise map
    print('    ... loading noise map')
    try:
        k_obj = nib.load(noise)
    except:
        print('')
        raise RuntimeError('ERROR: the 3D input file {} does not exist or is not in NIFTI format.'.format(noise))
    k_data = k_obj.get_fdata()
    k_size = k_data.shape
    k_size = np.array(k_size)
    k_header = k_obj.header
    k_affine = k_header.get_best_affine()
    k_dims = k_obj.shape
    if k_size.size!=3:
        print('')
        raise RuntimeError('ERROR: the 3D input file {} is not a 3D NIFTI.'.format(noise))
    if ( (np.sum(k_affine==m_affine)!=16) or (k_dims[0]!=m_dims[0]) or (k_dims[1]!=m_dims[1]) or (k_dims[2]!=m_dims[2]) ):
        print('')
        raise RuntimeError('ERROR: the header geometry of {} and {} do not match.'.format(dwi,noise))

    ### Rescale noise map to account for any previous signal averaging performed before the actual estimation of the noise map itself
    noisemap = k_data*float(nsa)

    ### Load b-values if provided (in this case, the code will assume that the spherical mean has to be computed)
    if(bv is not None):
        bv_exists = True
        try:
            barray = np.loadtxt(bv)
        except:
            raise RuntimeError('ERROR. Incorrect format for b-value file - remember that the b-value file is loaded with numpy.loadtxt()')
        if(barray.ndim!=1):
            raise RuntimeError('ERROR. The b-value file must store a 1D array of space-separated b-values in s/mm2. If your b-value file has only 1 element, just set bv=None')
        
        # Check that the number of measurements in the b-value file matches the number of measurements in the NIFTI file
        if(barray.size!=m_size[3]):
            raise RuntimeError('ERROR. The number of measurements in the input NIFTI {} does not match the number of volumes in the b-value file {}'.format(dwi,bv))            

    else:
        bv_exists = False
    
    ### Compute spherical mean if needed
    if(bv_exists):

        # Find unique number of b-values
        print('    ... computing spherical mean')
        b_unique = np.unique(barray)
        n_unique = b_unique.size
        sm_raw = np.zeros( (m_size[0],m_size[1],m_size[2],n_unique) )

        # Calculate spherical mean at fixed b-value
        for bb in range(0,n_unique):
            images = m_data[:,:,:,barray==b_unique[bb]]
            sm_raw[:,:,:,bb] = np.nanmean(images,axis=3)

    else:

        # The input NIFTI already contains spherical means
        sm_raw = np.copy(m_data)
        del m_data


    ### Now apply the unbiasing formula to get a better estimate of the spherical mean
    print('    ... unbiasing spherical mean')
    sm_unbias = np.zeros(sm_raw.shape)
    for bb in range(0,sm_raw.shape[3]):
        sm_bb = sm_raw[:,:,:,bb]
        sm_bb_unbias = sm_bb - 0.5*noisemap*noisemap/sm_bb
        sm_bb_unbias[np.isnan(sm_bb_unbias)] = 0.0
        sm_bb_unbias[np.isinf(sm_bb_unbias)] = 0.0
        sm_unbias[:,:,:,bb] = sm_bb_unbias


    ### Save out unbiased spherical mean signals
    print('    ... saving output files')
    m_header.set_data_dtype('float64')   # Make sure we save output files float64, even if input is not
    out_obj = nib.Nifti1Image(sm_unbias,m_affine,m_header)
    nib.save(out_obj,outfile)
    if(bv_exists):
       np.savetxt('{}.bval_sph_mean'.format(outfile),[b_unique],fmt='%.2f',delimiter=' ')
    

    ### Done
    print('    ... done')
    print('')
    print('')




# Run the module as a script when required
if __name__ == "__main__":

    ### Parse arguments or print help
    parser = argparse.ArgumentParser(description='Mitigate the noise floor bias in spherical mean diffusion-weighted signals using the analytical expression of Pieciak T. et al, Proc ISMRM 2022, p.3900 (SM_unbiased = SM - 0.5*sigma^2/SM). Author: Francesco Grussu, July 2023. Email: <fgrussu@vhio.net>. Third-party dependencies: numpy, nibabel. Developed with numpy 1.23.5 and nibabel 5.1.0.')
    parser.add_argument('dwi', help='path of a 4D NIFTI file storing diffusion MRI data. The NIFTI file can either i) already contain spherical mean data at fixed b-value, or ii) contain images acquired at different gradient directions; in this case, the spherical mean will be computed. The default is i); to flag that the input images corresponds to different gradient directions, pass a b-value list via option --bval (see below).')
    parser.add_argument('sigma', help='path of a 3D NIFTI file storing a voxel-wise noise map (standard deviation of Gaussian/Rician noise). This can be obained, for example, via Marchenko-Pastur Principal Component Analysis (Veraart J et al, NeuroImage 2016).')
    parser.add_argument('out', help='path of a 4D NIFTI file storing the unbiased spherical mean. If the input file contained images corresponding to different gradient directions, as flagged by passing a b-value file with option --bv, then the output file will have fewer volumes than the inut file, as it will contain spherical mean signals. In this case, additional text file, where "*.bval_sph_mean" is appended to the output file name, will be saved, to store the b-values of the spherical mean signals.')
    parser.add_argument('--bval', metavar='<file>', help='path a a text file storing b-values (FSL format; in s/mm2; one raw, space-separated). If provided, the code will assume that the input NIFTI file contains images corresponding to different gradient directions, so that the spherical mean needs calculating, before unbiasing can be performed. Spherical mean signals will be calcualated by averaging images with the same b-value. Default: None (no b-value list provided, implying that the input NIFTI file stores spherical mean signals)')
    parser.add_argument('--nsa', metavar='<num>', default=1, help='total number of signal averages performed on the image on which the noise map was calculated (default: --nsa 1). For example, if 1 b-value is acquired with NEX=2 using trace imaging (acquisition of 3 orthogonal gradients, each with NEX=2, which are then averaged by the scanner), then NSA = 6 (2 averages per gradients, with averaging of gradients performed by the scanner).')
    args = parser.parse_args()

    ### Get subset
    unbiasSM(args.dwi,args.sigma,args.out,nsa=args.nsa,bv=args.bval)
    sys.exit(0)
