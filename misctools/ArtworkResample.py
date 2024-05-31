### Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain). 
#   All rights reserved.
#   This code is distributed under the Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)).
#   Its use MUST also comply with the requirements of the individual licenses of all its dependencies.
#
# Resample an image to PNG or TIFF with a given size and DPI
#
# Author: Francesco Grussu <fgrussu@vhio.net>

import numpy as np
from PIL import Image
import argparse
Image.MAX_IMAGE_PIXELS = 933120000



if __name__ == "__main__":

	
	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='Resample an image to PNG or TIFF with a given size and DPI. Author: Francesco Grussu <fgrussu@vhio.net>.')	
	parser.add_argument('img_in', help='path of the image to resample')
	parser.add_argument('img_out', help='path of the output resampled image (it can be PNG or TIFF)')
	parser.add_argument('wout_inch', help='desired width of the output image (in inches)')
	parser.add_argument('dpi_out', help='desired resolution of output image (in dot-per-inch)')
	parser.add_argument('--compress', metavar='<VALUE>', help='compression. For TIFF, VALUE can be any of None, "group3", "group4", "jpeg", "lzma", "packbits", "tiff_adobe_deflate", "tiff_ccitt", "tiff_lzw", "tiff_raw_16", "tiff_sgilog", "tiff_sgilog24", "tiff_thunderscan", "webp", "zstd". For PNG, VALUE indicates the ZLIB compression level, and must be a number between 0 and 9: 1 gives best speed, 9 gives best compression, 0 gives no compression.  Default: no compression')	
	args = parser.parse_args()
	
	## Load image and get size
	din = Image.open(args.img_in)
	din_size = din.size
	win_pix = din_size[1]     # width of input image in pixels
	
	## Get size of output image
	wout_pix = int(float(args.wout_inch)*float(args.dpi_out))
	hout_pix = int(din_size[0]*(float(wout_pix)/float(win_pix)))
	
	## Resample
	dout = din.resize((hout_pix,wout_pix))
	
	## Save
	
	# No compression
	if args.compress is None:
		try:
			dout.save(args.img_out, dpi=( int(float(args.dpi_out)), int(float(args.dpi_out)) )  )
		except:
			raise RuntimeError('output image format is not understood.')
	
	# compression
	else:
	
		# Try saving as TIFF
		try:
			if args.compress=="tiff_jpeg":
				dout.save(args.img_out, dpi=( int(float(args.dpi_out)), int(float(args.dpi_out)) ) , compression=args.compress , quality=100 )
			else:
				dout.save(args.img_out, dpi=( int(float(args.dpi_out)), int(float(args.dpi_out)) ) , compression=args.compress )
		except:
		
			# Try saving as PNG
			try:
				dout.save(args.img_out, dpi=( int(float(args.dpi_out)), int(float(args.dpi_out)) ) , compression=args.compress )
			except:
				raise RuntimeError('output image format is not understood.')
			
	## Print some information
	print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
	print('                  ArtworkResample.py                 ')
	print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
	print('')
	print('* Input image: {}'.format(args.img_in))
	print('* Output image: {}'.format(args.img_out))
	print('* Output image width: {} inches'.format(args.wout_inch))
	print('* Output image resolution: {} DPI'.format(args.dpi_out))
	if args.compress is not None:
		print('* Output compression: {}'.format(args.compress))
	else:
		print('* Output compression: None')
	print('')
	print('   ... resampled ({},{}) --> ({},{})'.format(din_size[0],din_size[1],hout_pix,wout_pix))
	print('')
	
