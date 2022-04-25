# _*_ coding=utf-8 _*_

import os
import cv2 as cv2
import numpy as numpy


# Distance around a pixel
# ##################### #
# sqrt(2)   1   sqrt(2) #
#       1   0   1       #
# sqrt(2)   1   sqrt(2) #
# ##################### #

# now.sdf = lambed now : [(near.sdf + distance(now,near)) for near in aroundPx(now)].min


# How to generate a SDF

# now.sdf = 999999;
# if(now in object){
#     now.sdf = 0;
# }else{
#     foreach(near in nearPixel(now)){
#         now.sdf = min(now.sdf,near.sdf + distance(now,near));
#     }
# }



def distance(pPointA, pPointB):
    pass

