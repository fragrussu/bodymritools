This script can be called directly from the command line. Its input parameters are the following: 
```
usage: CreateCustomSchemeFile.py [-h] bval gdur gsep outscheme

Creates a scheme file for pgse2sandi.py for an acquisition with multiple b-values and one, fixed diffusion time.

positional arguments:
  bval        path a a text file storing b-values (FSL format; in s/mm2; one raw, space-separated)
  gdur        scalar indicating the gradient duration (in ms) used for the acquisition of all b-values
  gsep        scalar indicating the gradient separation (in ms) used for the acquisition of all b-values
  outscheme   output file storing the scheme file for SANDI fitting through pgse2sandi.py. It is a space-separated, FSL-like diffusion protocol file, with 3 rows and as many columns as diffusion
              images. -- First row: b-values in s/mm2; -- Second row: gradient duration small delta in ms; -- Third row: gradient separation Large Delta in ms.

options:
  -h, --help  show this help message and exit
```
