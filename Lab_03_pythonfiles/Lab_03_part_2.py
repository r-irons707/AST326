# part 2 - working with only the filtered files from filter 2

# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco # curve fitting
from astropy.io import fits # handeling FITS file
from astropy.visualization import astropy_mpl_style
from astropy.wcs.utils import skycoord_to_pixel
from matplotlib.ticker import MultipleLocator, AutoMinorLocator # axis mechanics
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob as glub # gathering files
from datetime import datetime # time sorting library for the fits files
from astropy.coordinates import SkyCoord,FK5 # coordinate conversions
import astropy.units as u # astropy units
from astropy.wcs import WCS
from Astro_functions import *

# centroid libraries
# citations
# © Copyright 2011-2022, Photutils Developers.
# Created using Sphinx 5.0.2.   Last built 13 Jul 2022.
import warnings

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Const1D, Const2D, Gaussian1D, Gaussian2D
from astropy.utils.exceptions import AstropyUserWarning
from photutils.centroids import centroid_1dg, centroid_2dg
__all__ = ['centroid_1dg', 'centroid_2dg']

# load in files
# loading and sorting files
filenameslist = np.genfromtxt('AST325-326-SN-filelist.txt',dtype='str')
filedata = glub('AST325-326-SN/*.fits')
# load in each of the fits files and extract the data for each
images = [] # empty images list
images_info = [] # empty header info 
file_times = []
for name in filedata:
    image_open = fits.open(name) # open fits file
    image_data = image_open[0].data # image data
    image_header = image_open[0].header # header info
    dt = datetime.fromisoformat(image_header["DATE-OBS"]) # find date time information from the header
    images.append(image_data) # append the image data
    images_info.append(image_header) # append the header info
    file_times.append(dt) # append the fits files times

# sort the three (data,headers,date time) lists by same method at same time
file_times,images = zip(*sorted(zip(file_times,images)))
file_times,images_info = zip(*sorted(zip(file_times,images_info)))
del file_times # delete file_times from memory

# convert all data to arrays
images = np.array(images)
images_info = np.array(images_info,dtype=object)

# determining pixel positions of all three reference stars  from the given hour angles
ref1 = ['00h56m49.70s -37:01:38.31']
ref2 = ['00h56m46.43s -37:02:29.50']
ref3 = ['00h56m58.27s -36:58:16.60']
# converting hours to RA and DEC
REF1 = SkyCoord(ref1,frame=FK5,unit=(u.hourangle,u.deg))
REF2 = SkyCoord(ref2,frame=FK5,unit=(u.hourangle,u.deg))
REF3 = SkyCoord(ref3,frame=FK5,unit=(u.hourangle,u.deg))
# [[ref 1 RA, ref 1 DEC], [ref 2 RA, ref 2 DEC], [ref 3 RA, ref 3 DEC]]
RA_DEC_list = np.array([[14.20708333,-37.02730833],[14.19345833,-37.04152778],[14.24279167,-36.97127778]])

# axe some initial images from the beginning of the file
print('len images (how many files we are dealing with): ',len(images))
# axe images [3,4,5,6,7,8,9] from the data set just from inspection of first 20 images
image_holder1 = images[:3]
image_holder2 = images[10:]
images = np.concatenate((image_holder1,image_holder2),axis=0)
# check the new result for removed images, delete image holder variables from memory
del image_holder1, image_holder2

# index the rest of the images and headers from the 2nd filter
indices = np.loadtxt('SN_indices.txt',delimiter=',',dtype=int)
print('prefilter length: ',len(images))
images = images[indices]
images_info = images_info[indices]
print('post filter length: ',len(images))

# filtered image look pretty solid, now we will do the centroid finding and everything on them (copy/paste .py part 1 code)
for i in range(5):
    # for a given image header, we want to convert the RA, DEC's into pixel position on the file
    pxx,pxy = pixel_positions(RA_DEC_list,images_info[i])

    # conversion looks good, now create a 2D box about the star pixel positions given an image
    # so we can narrow down exact positions, we will import this as a function
    box_reference_stars = box(pxx,pxy,150,images[i])

    # boxes look good, convert box sizes to 150px in diameter to make sure stars are within frame and the diameter of the annulus can fit
    # next step: find exact center of each star in a given box through 2D Gaussian fitting (require photutils)
    box_centroid = []
    for i in range(len(box_reference_stars)):
        box_centroid.append(centroid_2dg(box_reference_stars[i], error=None, mask=None))
    box_centroid = np.array(box_centroid,dtype=int)

    # we need to convert the pixel position returned from the box segment to the actual pixel location in the overall image
    # px pos on image = (distance to box edge) + (radius box_centroid)
    # = (old px pos) - (radius box_reference_stars) + (radius box_centroid)
    # note this is an array of three values
    pxx_centroid2image = np.array([pxx[0] - 75 + box_centroid[0][0],pxx[1] - 75 + box_centroid[1][0],\
            pxx[2] - 75 + box_centroid[2][0]], dtype=int)
    pxy_centroid2image = np.array([pxy[0] - 75 + box_centroid[0][1],pxy[1] - 75 + box_centroid[1][1],\
            pxy[2] - 75 + box_centroid[2][1]], dtype=int)
    print('original positions from given info: ',pxx,pxy)
    print('coverted px positions on the image from centroid position:',pxx_centroid2image,'y info:', pxy_centroid2image)

    # this conversion looks good, now we need to center the boxes about the centroid center, and not the reported RA/DEC's
    # call box function on the pixel positions, we can also delete the previous box info from memory
    del box_reference_stars, box_centroid, pxx, pxy
    centroid_sub_image = box(pxx_centroid2image,pxy_centroid2image,150,images[i])
    # calculate new center of the box, this will give us a better result than before
    box_centroid = []
    for i in range(len(centroid_sub_image)):
        box_centroid.append(centroid_2dg(centroid_sub_image[i], error=None, mask=None))
    box_centroid = np.array(box_centroid,dtype=int)
    # i believe its a good idea to perform the centroiding algorithm twice as some stars were skewed, resulting in non-optimal positions,
    # and stars not actually being at the center of the image, therefore we delete and repeat
    pxx_centroidtoimage = np.array([pxx_centroid2image[0] - 75 + box_centroid[0][0],pxx_centroid2image[1] - 75 + box_centroid[1][0],\
            pxx_centroid2image[2] - 75 + box_centroid[2][0]], dtype=int)
    pxy_centroidtoimage = np.array([pxy_centroid2image[0] - 75 + box_centroid[0][1],pxy_centroid2image[1] - 75 + box_centroid[1][1],\
            pxy_centroid2image[2] - 75 + box_centroid[2][1]], dtype=int)
    # and now we can use this pixel value for the box function a last time, and delete the box_centroid
    del box_centroid,centroid_sub_image
    centroid_sub_image = box(pxx_centroidtoimage,pxy_centroidtoimage,150,images[i])
    # and find the center/ peak position in the box
    px_centroid = []
    for i in range(len(centroid_sub_image)):
        px_centroid.append(centroid_2dg(centroid_sub_image[i], error=None, mask=None))
    px_centroid = np.array(px_centroid,dtype=int)
    # and reconvert this to the position on the overall image file
    pxx_convert = np.array([pxx_centroidtoimage[0] - 75 + px_centroid[0][0],pxx_centroidtoimage[1] - 75 + px_centroid[1][0],\
            pxx_centroidtoimage[2] - 75 + px_centroid[2][0]], dtype=int)
    pxy_convert = np.array([pxy_centroidtoimage[0] - 75 + px_centroid[0][1],pxy_centroidtoimage[1] - 75 + px_centroid[1][1],\
            pxy_centroidtoimage[2] - 75 + px_centroid[2][1]], dtype=int)
    print('coverted px positions on the image from centroid position:',pxx_centroidtoimage,'y info:', pxy_centroidtoimage)
    print('coverted px positions on image round 2:',pxx_convert,'y info:', pxy_convert)
    print('centroid data: ', px_centroid)

    # these look better after the second round, now we delete unnecessary variables for sake of space
    del pxx_centroid2image,pxy_centroid2image, pxx_centroidtoimage, pxy_centroidtoimage

    # apply gaussian fitting of each reference star so determine the standard deviation
    X,Y = [], []
    popt_norm,pcov_norm = [], []
    images_filterindex = []
    aperture = []
    for k in range(len(centroid_sub_image)): # compute std for each sub image to use for the mask
        X.append(np.arange(centroid_sub_image[k].shape[0]))
        Y.append(centroid_sub_image[k][px_centroid[k,1],:])
        popt_gauss, pcov_gauss = sco.curve_fit(gaus,X[k],Y[k],p0=[800,70,10,5])
        popt_norm.append(popt_gauss)
        pcov_norm.append(pcov_gauss)
        sigma = popt_norm[k][2]
        
        # now we will do the aperature mask for each of the reference stars
        aperture.append(get_aperture(centroid_sub_image[k],sigma))
    aperture = np.array(aperture)
    
    # plotting the resulting images and the pixel location on each file with the new centroid pixel
    fig,ax0 = plt.subplots(figsize=(10,7),ncols=1,nrows=1)

    im = ax0.imshow(images[i],cmap='gray',vmin=np.percentile(image_data[i],5),
                    vmax=np.percentile(image_data[i],99),origin='lower')
    #plt.scatter(pxx,pxy,c='r')
    #plt.scatter(pxx_centroid2image,pxy_centroid2image,c='c',marker='x')
    plt.scatter(pxx_convert,pxy_convert,c='r',marker='x')
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes('right',size='5%',pad=0.05)
    ax0.grid(False)
    fig.colorbar(im,cax=cax,orientation='vertical')
    ax0.set_title('file 0',fontsize=22)
    ax0.set_xlabel('Spatial direction',fontsize=22)
    ax0.set_ylabel('Spatial direction',fontsize=22)
    plt.tight_layout()

    # plot boxes and scattered centroid
    fig,((ax0,ax1,ax2),(ax3,ax4,ax5),(ax6,ax7,ax8)) = plt.subplots(figsize=(10,7),ncols=3,nrows=3)
    ax0.imshow(centroid_sub_image[0],cmap='gray',vmin=np.percentile(image_data[i],5),vmax=np.percentile(image_data[i],99),origin='lower')
    ax0.scatter(px_centroid[0][0],px_centroid[0][1],c='r')
    ax1.imshow(centroid_sub_image[1],cmap='gray',vmin=np.percentile(image_data[i],5),vmax=np.percentile(image_data[i],99),origin='lower')
    ax1.scatter(px_centroid[1][0],px_centroid[1][1],c='r')
    ax1.set_title('located reference stars image')
    ax2.imshow(centroid_sub_image[2],cmap='gray',vmin=np.percentile(image_data[i],5),vmax=np.percentile(image_data[i],99),origin='lower')
    ax2.scatter(px_centroid[2][0],px_centroid[2][1],c='r')
    # plot std of each star under its corresponding image box with center data slice and gaussian fitting
    ax3.plot(X[0],gaus(X[0],*popt_norm[0]),color = 'r', alpha = 0.3, label = f'Best-fit Gaussian (STD {round(popt_norm[0][2],1)})')
    ax3.scatter(X[0],Y[0])
    ax3.legend()
    ax4.plot(X[1],gaus(X[1],*popt_norm[1]),color = 'r', alpha = 0.3, label = f'Best-fit Gaussian (STD {round(popt_norm[1][2],1)})')
    ax4.scatter(X[1],Y[1])
    ax4.legend()
    ax4.set_title('center slice and Gaussian fitting image')
    ax5.plot(X[2],gaus(X[2],*popt_norm[2]),color = 'r', alpha = 0.3, label = f'Best-fit Gaussian (STD {round(popt_norm[2][2],1)})')
    ax5.scatter(X[2],Y[2])
    ax5.legend()
    # plot apertures and annulus masks
    ax6.imshow(aperture[0]*centroid_sub_image[0],origin='lower',cmap='gray',vmin=np.percentile(image_data[i],5),vmax=np.percentile(image_data[i],99))
    ax7.imshow(aperture[1]*centroid_sub_image[1],origin='lower',cmap='gray',vmin=np.percentile(image_data[i],5),vmax=np.percentile(image_data[i],99))
    ax8.imshow(aperture[2]*centroid_sub_image[2],origin='lower',cmap='gray',vmin=np.percentile(image_data[i],5),vmax=np.percentile(image_data[i],99))

    plt.tight_layout()
    plt.show()