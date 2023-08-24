import numpy as np
from imantics import Polygons, Mask
import cv2, os

path = "/home/ise.ros/akshay_work/datasets/tless_mod/tless_mod/train_pbr/000001/mask"
img_name = "000000_"
files = os.listdir(path)
req_files = [os.path.join(path,a) 
             for a in files if img_name in a]

all_arr = np.zeros((360,640))
for f in req_files:
    img = cv2.imread(f)
    #array = np.ones((100, 100))
    array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_arr += array

    polygons = Mask(array).polygons()

    #print(polygons.points)
    print(len(polygons.segmentation[0]))

cv2.imshow("all_mask", all_arr)
cv2.waitKey(0)