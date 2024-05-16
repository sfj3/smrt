import cv2, sys, os
import numpy as np
import math
from scipy.linalg import norm
from math import atan
from math import copysign, log10
import scipy.spatial
from scipy.spatial import procrustes
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize,thin
from skimage import data
from skimage.util import invert
def main():
    # Obtain filenames from command line argument
    filename = sys.argv[1]
    img = Image.open(filename)
    im = cv2.imread(filename,0)
    #skeletonize input    
    imageinverse = im
    for i in range(len(im)):
        for j in range(len(im[i])):
            if im[i][j]!=255:
                im[i][j]=1.0
                imageinverse[i][j]=0

            else:
                im[i][j]=0.0
                imageinverse[i][j]=1.0
    im = thin(im)
    
    filename1 = sys.argv[2]
    img1 = Image.open(filename1)
    im1 = cv2.imread(filename1,0)
    image1inverse = im1
    for i in range(len(im1)):
        for j in range(len(im1[i])):
            if im1[i][j]!=255:
                im1[i][j]=1.0
                image1inverse[i][j]=0
            else:
                im1[i][j]=0.0
                image1inverse[i][j]=1.0
    im1 = thin(im1)
    imageinverse = thin(imageinverse)
    image1inverse = thin(image1inverse)
    #get x y coordinates of image
    coordinates = []
    coordinates1 = []
    #lets add every couple of points
    #image resized 5% from original in paint
    xs = []
    ys = []
    xs1=[]
    xs2=[]
    for i in range(0,len(im),1):
        for j in range(0,len(im[i]),1):
            if im[i][j]!=0:
                    coordinates.append([i,j])
                    xs.append(i)
                    ys.append(j)
    for i in range(0,len(im1),1):
        for j in range(0,len(im1[i]),1):
            if im1[i][j]!=0:
                    coordinates1.append([i,j])
                    xs1.append(i)
                    xs2.append(j)
    impoPts = []
    impoPts1=[]
    zerocounter=0

    #try ad plot only important points
    xim = []
    yim = []
    prev = 0
    firstpt = [0,0]
    scale = len(coordinates)/len(coordinates1)
    for i in range(0,len(coordinates),int(len(coordinates)/100)):#his limites the number of points
        #dslope = (coordinates[i][1]-firstpt[1]) * (coordinates[i][0]-firstpt[0])
        ##only take the point if there is a significant change in the slope
        #print(dslope)
        #if abs(dslope)>2 or abs(dslope)<0.5:
        firstpt = coordinates[i]
        xim.append(coordinates[i][0])
        yim.append(coordinates[i][1])
        impoPts.append(coordinates[i])

    
        #dslope = (coordinates[i+5][1]-coordinates[i][1])/(coordinates[i+5][0]-coordinates[i][0]) - (coordinates[i][1]-coordinates[i][1])/(coordinates[i-5][0]-coordinates[i-5][0])
    
    xim1 =[]
    yim1=[]
    firstpt=[0,0]
    for i in range(0,len(coordinates1),int(len(coordinates1)/100)):
        #dslope = (coordinates1[i][1]-firstpt[1])+(coordinates1[i][0]-firstpt[0])
        #print(dslope)
        #if abs(dslope)>2:
        #firstpt = coordinates[i]
        xim1.append(coordinates1[i][0])
        yim1.append(coordinates1[i][1])
        impoPts1.append(coordinates1[i])
        #dslope = (coordinates[i+5][1]-coordinates[i][1])/(coordinates[i+5][0]-coordinates[i][0]) - (coordinates[i][1]-coordinates[i][1])/(coordinates[i-5][0]-coordinates[i-5][0])

    impoa = np.array(impoPts)
    impoa1 = np.array(impoPts1)
    mean = np.mean(impoa, axis=0)
    #these means are the translation
    for i in impoa:
        i[0]-=mean[0]
        i[1]-=mean[1]
    mean1 = np.mean(impoa1,axis=0)
    for i in impoa1:
        i[0]-=mean1[0]
        i[1]-=mean1[1]
    if len(impoa)<len(impoa1):
        for i in range(len(impoa1)-len(impoa)):
            impoa = np.append(impoa,np.array([[0,0]]),axis=0)

    if len(impoa1)<len(impoa):
        for i in range(len(impoa)-len(impoa1)):
            impoa1 = np.append(impoa1,np.array([[0,0]]),axis=0)
    qe,a2fits1,x = scipy.spatial.procrustes(impoa,impoa1)
    a,b = scipy.linalg.orthogonal_procrustes(impoa,impoa1)
    c,d = scipy.linalg.orthogonal_procrustes(impoa,impoa)
    ot,de = scipy.linalg.orthogonal_procrustes(impoa1,a2fits1)

    
    cbcb,bcba,xx = scipy.spatial.procrustes(image1inverse,imageinverse)
    print("Similarity")
    valset = x
    print(valset)
    #mean_y = np.mean(shape[1::2]).astype(np.int)
    #a,b,c =  scipy.spatial.procrustes(impoPts, impoPts)
    #a1,b1,c1 = scipy.spatial.procrustes(impoPts, impoPts1)
    #print(a,b,c)
    print("magnitude scale")
    print(scale)
    print('rotation angle')
    angle = -180/3.14159*math.asin(a[0][1])
    print(angle)
    print('translation [x,y] in mm')
    mean1[0]-=mean[0]
    mean1[1]-=mean[1]
    print(mean1*0.26)
    print()
    #to show any plot its plt.scatter(xim,yim) <- you make those yourself
    #plt.show()
if __name__ == "__main__":
    main()