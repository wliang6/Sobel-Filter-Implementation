CSE 473 Computer Vision and Image Processing
June 29th, 2016
Author: Winnie Liang

Before running the code, ensure you have OpenCV library installed onto your Python libraries. Enthought Canopy IDE and Python 2.7 is used and recommended.


Input images are provided and are located in the same directory as the code.   
    'rose.jpg' 'lena_gray.jpg'

The files below is implemented using the CV2 library by calling filter2D and sepFilter2D functions.
    '1Dconvolution.py' '2Dconvolution.py'

The files below is my own custom implementation of image convolution using Sobel Filters.
    'custom1Dconvolution.py' 'custom2Dconvolution.py'



To run the code on the terminal:
	To perform 1D Convolution using Sobel filters, type in:
        ------python 1Dconvolution.py
                    OR
        ------python custom1Dconvolution.py

	To perform 2D Convolution using Sobel filters, type in:
        ------python 2Dconvolution.py
                    OR
        ------python custom2DConvolution.py

	To perform Histogram Equalization, type in:
        ------python histogram.py

