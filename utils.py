import colorsys
from os import replace
import random

import numpy as np
import pandas as pd
import skimage.draw as drw

# dataframe prep
def _fix_df(df):
    """Prepare the Data Frame to be readable
    """
    df_new = df.drop(['ID'], axis=0)
    df_new.columns = df_new.iloc[0,:]
    df_new.drop([np.nan], axis=0, inplace=True)
    df_new.columns.name = 'ID'
    return df_new
    
"""CNN prep and analysis"""

def crop(image, xs, xe, ys, ye):
    return image[ys:ye, xs:xe]

# crop image. use a bounding box of certain size greater than the size of the optic disc
def crop_image_from_contour(image, contour, imagedim):
    sumx=0
    sumy=0
    count=0
    for coord in contour:
        sumx = sumx + coord[0]
        sumy = sumy + coord[1]
        count= count+1

    # this is a method to find a quick estimate for a midpoint for the optic disc.
    # Then we can use this, to create a bounding box of a certain size around this point. 
    # The size will be the same for all the other images. 
    # i will be using an image of size 1000x1000
    middlex = sumx//count
    middley = sumy//count

    imagedimDividedBy2 = imagedim//2  #reduce computation

    startx = int(middlex - imagedimDividedBy2)
    endx = int(middlex + imagedimDividedBy2)
    starty = int(middley - imagedimDividedBy2)
    endy = int(middley + imagedimDividedBy2)

    return crop(image,startx, endx,starty,endy)


# contour_to_mask() and apply_mask() functions taken from https://github.com/matterport/Mask_RCNN
def contour_to_mask(cont, img_shape, abs_path='../'):
    """Return mask given a contour and the shape of image
    """
    c = np.loadtxt(abs_path + "ExpertsSegmentations/Contours/" + cont)
    mask = np.zeros(img_shape[:-1], dtype=np.uint8)
    rr, cc = drw.polygon(c[:,1], c[:,0])
    mask[rr, cc] = 1
    return mask

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image



""" clinical data options analysis"""

def read_clinical_data(abs_path='../'):
    """Return excel data as pandas Data Frame
    """
    df_od = pd.read_excel(abs_path + 'ClinicalData/patient_data_od.xlsx', index_col=[0])
    df_os = pd.read_excel(abs_path + 'ClinicalData/patient_data_os.xlsx', index_col=[0])
    return _fix_df(df=df_od), _fix_df(df=df_os)

def get_diagnosis(abs_path='../'):
    """Return three arrays of shape 488 with the diagnosis tag, eye ID (od, os)
    and patient ID
    """
    df_od, df_os = read_clinical_data(abs_path=abs_path)
        
    index_od = np.ones(df_od.iloc[:,2].values.shape, dtype=np.int8)
    index_os = np.zeros(df_os.iloc[:,2].values.shape, dtype=np.int8)

    eyeID = np.array(list(zip(index_od, index_os))).reshape(-1)
    tag = np.array(list(zip(df_od.iloc[:,2].values, df_os.iloc[:,2].values))).reshape(-1)
    patID = np.array([[int(i.replace('#', ''))] * 2 for i in df_od.index]).reshape(-1)
    
    return tag, eyeID, patID