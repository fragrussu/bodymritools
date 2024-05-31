### Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain). 
#   All rights reserved.
#   This code is distributed under the Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)).
#   Its use MUST also comply with the requirements of the individual licenses of all its dependencies.
#
#   Convert a .seg.nrrd segmentation to NIFTI, borrowing the header from a reference NIFTI file
#   AUTHOR
#   Francesco Grussu, Vall d Hebron Institute of Oncology, Barcelona (Spain)
#   Email: <fgrussu@vhio.net>
#

import argparse
import sys
import csv
import nrrd
import numpy as np
import nibabel as nib

def convert(segfile, reffile, outfile, outcsv):
	'''
	   Convert a .seg.nrrd segmentation to NIFTI, borrowing the header from a reference NIFTI file
	   
	   USAGE
	   convert(segfile, reffile, outfile, outcsv)
	   
	   INPUT ARGUMENTS
	   * segfile: Segmentation file in NRRD format (.seg.nrrd, as those provided by Slicer)
	   * reffile: Reference scan in NIFTI format (a.k.a. "master scan" using Slicer nomenclature)
	   * outfile: Output file storing the segmentation converted to NIFTI format
	   * outcsv:  Output file storing the segmentation labels as a CSV file
	   
	   AUTHOR
	   Francesco Grussu, Vall d Hebron Institute of Oncology, Barcelona (Spain), April-May 2021
	   Email: <francegrussu@gmail.com> <fgrussu@vhio.net>
	   
	'''


	# Print some feedback
	print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
	print('                      segnrrd2nii.convert()                             ')
	print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
	print('')
	print('* Segmentation file in NRRD format: {}'.format(segfile))
	print('* Reference scan in NIFTI format: {}'.format(reffile))
	print('* Output file storing the segmentation converted to NIFTI format: {}'.format(outfile))
	print('* Output file storing the segmentation labels as a CSV file: {}'.format(outcsv))
	print('')

	# Load NRRD and NIFTI files
	print('')
	print('         ... loading data')
	print('')
	nrrd_data, nrrd_header = nrrd.read(segfile)  # Get NRRD data and header
	nifti_obj = nib.load(reffile)       # Load NIFTI to Nibabel object
	nifti_header = nifti_obj.header     # Get NIFTI header
	nifti_affine = nifti_obj.affine     # Get NIFTI scanner-to-image transformation
	nifti_data = nifti_obj.get_fdata()  # Get NIFTI data to check image dimensions
	
	# Convert NRRD data to floating point
	nrrd_data = np.array(nrrd_data,dtype='float64')
	
	# Save the NRRD as a NIFTI
	if ( (nifti_data.shape[0]!=nrrd_data.shape[0]) or (nifti_data.shape[1]!=nrrd_data.shape[1]) or (nifti_data.shape[2]!=nrrd_data.shape[2]) ):
		print('')
		print('######################################################################################')
		print('                                       WARNING                                        ')
		print('The dimensions of ')		
		print('{}'.format(segfile))
		print('and')
		print('{}'.format(reffile))
		print('do not match')
		print('######################################################################################')
		print('')
		print('         ... no output NIFTI file saved')
		print('')
	else:	
		print('')
		print('         ... converting and saving')
		print('')
		nifti_header.set_data_dtype('float64')
		out_obj = nib.Nifti1Image(nrrd_data,nifti_affine,nifti_header)
		nib.save(out_obj, outfile)
	
	
	# Extract segmentation information for the CSV file
	idx = 0
	row_list = [['Segment_ID','Segment_Name','Segment_Value']]
	while(True):
		
		# Check whether there is yet another segment
		try:

			# Yes, there is a new segment to process

			# Get segment ID, name and value 
			myid = nrrd_header['Segment{}_ID'.format(idx)]
			myname = nrrd_header['Segment{}_Name'.format(idx)]
			myvalue = myid.split('_')[1]
			myvalue = float(myvalue)

			# Store the information
			row_list.append([myid,myname,myvalue])
			
		except:

			# If not, terminate loop
			break

		idx = idx + 1


	# Save segmentation labels
	h = open(outcsv, 'w', newline='')
	writer = csv.writer(h)
	writer.writerows(row_list)
	h.close()
		

	# Done
	print('')
	print('         ... done')
	print('')

	
	

if __name__=="__main__":

	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='Convert a .seg.nrrd segmentation to NIFTI, borrowing the header from a reference NIFTI file. Author: Francesco Grussu, Vall d Hebron Institute of Oncology, Barcelona (Spain), April-May 2021. Email: <francegrussu@gmail.com> <fgrussu@vhio.net>.')
	parser.add_argument('seg', help='Segmentation file in NRRD format (.seg.nrrd, as those provided by Slicer)')
	parser.add_argument('ref', help='Reference scan in NIFTI format (a.k.a. "master scan" using Slicer nomenclature)')
	parser.add_argument('out', help='Output file storing the segmentation converted to NIFTI format')
	parser.add_argument('csv', help='Output file storing the segmentation labels as a CSV file')
	args = parser.parse_args()
	
	### Run conversion
	convert(args.seg, args.ref, args.out, args.csv)
	
	### Done
	sys.exit(0)
	
	
