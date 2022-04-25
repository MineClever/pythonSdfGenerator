# _*_ coding=utf-8 _*_

import os
import cv2 as cv2
import numpy as np


# Distance around a pixel
# ##################### #
# sqrt(2)   1   sqrt(2) #
#       1   0   1       #
# sqrt(2)   1   sqrt(2) #
# ##################### #

# now.sdf = lambed now : [(near.sdf + distance(now,near)) for near in aroundPx(now)].min



# ref to https://zhuanlan.zhihu.com/p/337944099
# How to generate a SDF

# now.sdf = 999999;
# if(now in object){
#     now.sdf = 0;
# }else{
#     foreach(near in nearPixel(now)){
#         now.sdf = min(now.sdf,near.sdf + distance(now,near));
#     }
# }




class GlobalVar ():
    mWidth = 256
    mHeight = 256



class Point () :
    
    def __init__ (self , pX ,pY):
        self.mDx = pX
        self.mDy = pY

    def DistSqrt (self):
        return self.__mDx**2 + self.__mDy**2

class Grid ():
    # set a grid which include a Point matrix
    def __init__ (self, pWidth:int , pHeight:int):
        self.mGrid = [[Point(0,0)]*pWidth]*pHeight
        for x in range(pWidth):
            for y in range(pHeight):
                self.mGrid[pWidth][pHeight] = Point(pWidth, pHeight)

lInside = Point(0,0)
lEmpty = Point(9999,9999)
lGridA = Grid ()
lGridB = Grid ()

def getPoint (pGrid : Grid , pX: int, pY: int):
    if (pX >=0 and pY >=0 and pX < GlobalVar.mWidth and pY < GlobalVar.mHeight):
        return pGrid.mGrid[pX][pY]
    else :
        return lEmpty
    
def putPoint (pGrid : Grid , pX: int, pY: int , pPoint :Point):
    pGrid.mGrid[pX][pY] = pPoint

def comparePoint (pGrid : Grid , pPoint:list[Point], pX: int, pY: int , pOffsetX:int , pOffsetY :int):
    lAroundPoint = getPoint(pGrid, pX+pOffsetX , pY+pOffsetY)
    lAroundPoint.mDx += pOffsetX
    lAroundPoint.mDy += pOffsetY

    if (lAroundPoint.DistSqrt() < pPoint.DistSqrt()):
        pPoint[0] = lAroundPoint