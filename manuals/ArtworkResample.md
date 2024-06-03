This script can be called soley from the command line. Its input parameters are the following: 
```
usage: ArtworkResample.py [-h] [--compress <VALUE>] img_in img_out wout_inch dpi_out

Resample an image to PNG or TIFF with a given size and DPI. Author: Francesco Grussu
<fgrussu@vhio.net>.

positional arguments:
  img_in              path of the image to resample
  img_out             path of the output resampled image (it can be PNG or TIFF)
  wout_inch           desired width of the output image (in inches)
  dpi_out             desired resolution of output image (in dot-per-inch)

options:
  -h, --help          show this help message and exit
  --compress <VALUE>  compression. For TIFF, VALUE can be any of None, "group3", "group4", "jpeg",
                      "lzma", "packbits", "tiff_adobe_deflate", "tiff_ccitt", "tiff_lzw",
                      "tiff_raw_16", "tiff_sgilog", "tiff_sgilog24", "tiff_thunderscan", "webp",
                      "zstd". For PNG, VALUE indicates the ZLIB compression level, and must be a
                      number between 0 and 9: 1 gives best speed, 9 gives best compression, 0 gives
                      no compression. Default: no compression
```
