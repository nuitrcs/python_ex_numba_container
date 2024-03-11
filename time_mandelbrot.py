##########################################################################
# time_mandelbrot.py
# 
# J. Giannini, March 2024
#
# Adapted from: https://numba.pydata.org/numba-doc/dev/user/examples.html
#
# Plots Mandelbrot fractal using just-in-time compiled functions with 
# Numba. Added more timing functions and some print statements for
# demonstration purposes. Also added try-except so script exits if 
# Numba cannot be loaded. Uses matplotlib backend 'agg' so 
# doesn't try to display a plot on the screen (for use on Quest).
#
# Usage: python time_mandelbrot.py
#
##########################################################################

## Import some stuff 
import sys
import os
from timeit import default_timer as timer
import numpy as np

## Print some information and exit if we don't have numba
print('Python version, OS : ')
print(sys.version)

try: 
    from numba import jit
except ImportError:
    print('\nI do not have numba - Exiting...')
    sys.exit()

print('\nI have numba')


## Import matplotlib
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

##########################################################################

@jit(nopython=True)
def mandel(x, y, max_iters):
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    i = 0
    c = complex(x,y)
    z = 0.0j
    for i in range(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i

    return 255

##########################################################################

@jit(nopython=True)
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color

    return image

##########################################################################

## Create fractal image and time successive function calls

## Note that we have to code in a kind of C-ish way for the @jit 
## decorator to work (creating the empty array ahead of time, and 
## passing it to the function which will populate it. 
 
image = np.zeros((500 * 2, 750 * 2), dtype=np.uint8)

## First function call
start_1 = timer()
create_fractal(-2.0, 1.0, -1.0, 1.0, image, 100)
end_1 = timer()

## Second function call
image *= 0 ## re-initialize the image
start_2 = timer()
create_fractal(-2.0, 1.0, -1.0, 1.0, image, 100)
end_2 = timer()


## Third function call
image *= 0 ## re-initialize the image
start_3 = timer()
create_fractal(-2.0, 1.0, -1.0, 1.0, image, 100)
end_3 = timer()


print('Time first function call : ', end_1 - start_1)
print('Time second function call : ', end_2 - start_2)
print('Time third function call : ', end_3 - start_3)


## Save the image

## get my home directory with os
my_home = os.path.expanduser('~')

fig, ax = plt.subplots()
im = ax.imshow(image)
ax.set_title('Mandelbrot fractal')

print('Saving figure to your home directory.')
plt.savefig(my_home + '/mandelbrot_plot.png', dpi = 200)





