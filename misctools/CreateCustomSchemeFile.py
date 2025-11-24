### Extract a subset a DW images given a specificed maximum b-values
import numpy as np
import sys, argparse


# Run the module as a script when required
if __name__ == "__main__":

	### Parse arguments or print help
	parser = argparse.ArgumentParser(description='text')
	parser.add_argument('bval', help='path a a text file storing b-values (FSL format; in s/mm2; one raw, space-separated)')
	parser.add_argument('gdur', help='scalar indicating the gradient duration (in ms) used for the acquisition of all b-values')
	parser.add_argument('gsep', help='scalar indicating the gradient separation (in ms) used for the acquisition of all b-values')
	parser.add_argument('outscheme', help='output file storing the scheme file for SANDI fitting through pgse2sandi.py. It is a space-separated, FSL-like diffusion protocol file, with 3 rows and as many columns as diffusion images. -- First row: b-values in s/mm2; -- Second row: gradient duration small delta in ms; -- Third row: gradient separation Large Delta in ms.')
	args = parser.parse_args()

	### Print information
	print('')
	print('***********************      CreateCustomSchemeFile.py       ***********************')
	print(' ')
	print('   ++ Input b-value file: {}'.format(args.bval))
	print('   ++ Gradient duration: {} ms'.format(args.gdur))
	print('   ++ Gradient separation: {} ms'.format(args.gsep))
	print('   ++ Output scheme: {}'.format(args.outscheme))
	print(' ')

	### Create scheme file for SANDI
	gdur = float(args.gdur)
	gsep = float(args.gsep)
	bvalarray = np.loadtxt(args.bval)
	gdurarray = gdur + 0.0*bvalarray
	gdurarray[bvalarray==0] = 0
	gseparray = gsep + 0.0*bvalarray
	gseparray[bvalarray==0] = 0
	schemearray = np.zeros((3,bvalarray.size))
	schemearray[0,:] = bvalarray
	schemearray[1,:] = gdurarray
	schemearray[2,:] = gseparray
	np.savetxt(args.outscheme, schemearray, fmt='%.2f', delimiter=' ')

	### Exit
	print('     .... scheme file created')
	print(' ')
	sys.exit(0)


