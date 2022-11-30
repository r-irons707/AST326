# functions for AST326 Lab 03
import numpy as np
from astropy.wcs import WCS

# error analysis and plotting methods
# define chi square function
def chi_square(y_measured, y_expected,errors):
    return np.sum( np.power((y_measured - y_expected),2) / np.power(errors,2) )

# define chi square reduced
def chi_square_reduced(y_measured,y_expected,errors,number_parameters):
    return chi_square(y_measured,y_expected,errors)/(len(y_measured - number_parameters))

# obtaining pixel positions of the reference stars
def pixel_positions(RA_DEC_array,header):
    '''obtain the x,y pixel positions for reference stars given RA and DEC in a given image with header information
    input:
        RA_DEC_array: an array of the right ascensions and declinations in order of number of reference stars.
        image: the images that contain each reference star
        header: the file information which contains coordinate transformations
    output:
        pxx,pxy: the x,y coordinate of the pixel position on the FITS files as read from the header'''
    # define pixel positions based on the length of reference star info
    pxx = np.zeros(len(RA_DEC_array),dtype=int)
    pxy = np.zeros(len(RA_DEC_array),dtype=int)
    wcs = WCS(header) # open the header and determine coordinate positions/conversions
    for i in range(len(RA_DEC_array)): # loop over the number of reference stars given
        x,y = wcs.all_world2pix(RA_DEC_array[i][0],RA_DEC_array[i][1],0) # 0 indicates the language is python
        pxx[i] = pxx[i] + int(x) # add each coodinate to the zeros array, [ref1 x, ref 2 x, ref 3 x]
        pxy[i] = pxy[i] + int(y) # [ref 1 y, ref 2 y, ref 3 y]
    del x,y
    return (pxx,pxy)

# create a box about a given pixel position
def box(pos_pxx,pos_pxy,length,image):
    '''this function creates a symetric box about a given position for each reference star in an image file
    input:
        pxx_pos,pxy_pos: 2D array given pixel direction in x,y
        length: specified box width size
    output:
        box: 2D array centered about given pixel position'''
    box = []
    for i in range(len(pos_pxx)):
        sub_im = image[int(pos_pxy[i])-length//2:int(pos_pxy[i])+length \
                //2-1,int(pos_pxx[i])-length//2:int(pos_pxx[i])+length//2-1]
        box.append(sub_im)
    box = np.array(box)
    del sub_im
    return box

# define a gaussian function to get the standard deviation of the star
def gaus(x,a,x0,sigma, c):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + c

# get the index of an image that meets the std requirements to be kept
def get_index(std_list):
    '''this functions p.urpose is to return true/false depending if the standard deviations of a reference star are outside a desired condition'''
    for i in range(len(std_list)):
        if (np.absolute(std_list[i]) <= 0) or (np.absolute(std_list[i]) >= 15):
            return False
        else:
            return True
        
def circle(x,y,center):
    '''defines a circle
    x,y: pixel lengths
    center: center point of the circle'''
    return (x-center)**2 + (y-center)**2
        
# use a binary mask to determine the inner annulus for a given star and std
def get_aperture(star,sigma):
    '''defines a 'circular' aperture about a gaussian source for a given sigma through a 2D mask of binary 0's and 1's representing the existence of an aperture
    input:
        star: given source
        signma: standard deviation for signal
        center: intake the center pixels (x,y) of the signal from a 2D gaussian fit to base the center off of
    output:
        mask: 2D array of zeros and ones representing the aperture with radius 3*sigma'''
    print('sigma in the function is: ', sigma)
    domain = np.copy(star) # copy the shape of the star's domain such that we will have a 2D array the same size
    print('domain length, domain[0] shape', len(domain), domain.shape)
    center = len(domain)/2 # define the center of the sub-image
    for i in range(len(domain)): # find the distance from the center for each pixel
        for j in range(len(domain)):
            pixel = circle(i,j,center)
            if pixel > 6*sigma:
                domain[i,j] = 0
            if pixel <= 6*sigma:
                domain[i,j] = 1
    return domain