"""Analyzes the (pre-extracted) Nuclei of 
K. Sirinukunwattana, S.E.A. Raza, Y.W Tsang, I.A. Cree, D.R.J. Snead, N.M. Rajpoot, 
Locality Sensitive Deep Learning for Detection and Classification of Nuclei in Routine Colon Cancer Histology Images
IEEE Transactions on Medical Imaging, 2016 (in press)

Using Lasagne and Theano setup
1. Fully Supervised Learning for CNN"""

from scipy import misc
import numpy as np 
import scipy.io as sio


def read_data():
    images = np.zeros((100,500,500,3), dtype=np.float32)
    path = "/home/veda/ColonHistology/CRCHistoPhenotypes_2016_04_28/Classification/"
    for i in range(1,100):
        filename = "img" + str(i)
        fullpath = path + filename + "/" + filename + ".bmp" 
        images[i-1,:,:,:] = misc.imread(fullpath)
    return images
    
images = read_images()