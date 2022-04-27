# _*_ coding=utf-8 _*_


import os, sys
import cv2 as cv2
import numpy as np

# def type
class Vector2 ():
    def __init__(self, x=0,y=0):
        self.data = np.zeros(2, dtype=float)
        self.x = x
        self.y = y

    @property
    def x (self):
        return self.data[0]

    @x.setter
    def x (self, value):
        self.data[0] = float(value)
        
    @property
    def y (self):
        return self.data[1]

    @x.setter
    def y (self, value):
        self.data[1] = float(value)

    def __add__ (self, value):
        return Vector2((self.x+value.x),(self.y+value.y))

    def length_squared (self) -> float:
        return (self.x*self.x + self.y*self.y)

    def length (self) -> float:
        return np.sqrt(self.length_squared())
    
# For a bitmap with representation:
# [0][0][0]
# [0][1][1]
# [1][1][1]

# using relative offset [offset x, offset y] :
# [-1,-1][0,-1][1,-1]
# [-1, 0][0, 0][1, 0]
# [-1, 1][0, 1][1, 1]

class SSEDT8 (object):

    class Grid ():
        
        def __init__(self, width:int , height:int):  
            self.width = width
            self.height = height
            self.size = Vector2(width,height)
            self.distances = [Vector2()]* (width*height)
        
        def has (self, x:int, y:int) -> bool:
            return (0 <= x and x < self.size.x and 0 <= self.size.y and y < self.size.y)

        def _index (self, x:int, y:int) -> int:
            return (y * self.size.x + x)


        def get_size(self) -> Vector2:
            return self.size

        def get_dist(self, x:int, y:int) -> Vector2:
            return self.distances[self._index(x,y)]

        def set_dist(self, x:int, y:int, p_dinstance:Vector2) :
            self.distances[self._index(x,y)] = p_dinstance

        def update (self, x:int, y:int, offset:Vector2):
            pos = Vector2(x,y)
            offset_pos = pos + offset
            distance = self.get_dist(x, y)
            dist_sq = distance.length_squared()
            
            if (self.has(offset_pos.x, offset_pos.y)):
                offset_dist = self.get_dist(offset_pos.x, offset_pos.y) + offset
                offset_sq = offset_dist.length_squared()
                if (offset_sq < dist_sq):
                    self.set_dist(x, y, offset_dist)


    @classmethod
    def apply_offsets (cls, p_grid : Grid,x :int ,y :int, p_offsets:list):
        size = p_offsets.size
        for i in range(size):
            p_grid.update(x,y,p_offsets[i])
    
    @classmethod
    def apply_pass (cls, p_grid : Grid, p_offsets1 : list, p_offsets2 : list, inverted=False):
        grid_size = p_grid.get_size()
        width = grid_size.x
        height = grid_size.y
        if (inverted):
            y = height - 1
            x = width - 1
            while (y > 0):
                while (x >= 0):
                    cls.apply_offsets(p_grid, x, y, p_offsets1)
                    x -= 1
                else:
                    x = 0
                while (x < width):
                    cls.apply_offsets(p_grid, x, y, p_offsets2)
                    x += 1
                y -= 1
                    
        

    @staticmethod
    def _bind_methods():
        pass

    @classmethod
    def do_sdf (cls, p_input_image_path='',p_output_image_path='', scale = 0.005):
        # read img by openCV
        img = cv2.imread(p_input_image_path)
        width = img.shape[0]
        height = img.shape[1]

        # Initialise grids
        grid1 = cls.Grid(width,height)
        grid2 = cls.Grid(width,height)
        DISTANT = sys.maxsize

        for y in range(height):
            for x in range(width):
                distance = 0 if img[x][y][2] > 0 else DISTANT
                grid1.set_dist(x, y, Vector2(distance, distance))
                grid2.set_dist(x, y, Vector2(DISTANT - distance, DISTANT - distance))


        # using relative offset [offset x, offset y] :
        # [-1,-1][0,-1][1,-1]
        # [-1, 0][0, 0][1, 0]
        # [-1, 1][0, 1][1, 1]
        
        # Pass 1

        # [2] [1] [ ]
        # [0] [ ] [ ]
        # [3] [ ] [ ]
        offsets1 = list()
        offsets1.append(Vector2(-1, 0))     # 0
        offsets1.append(Vector2(0, -1))     # 1
        offsets1.append(Vector2(-1, -1))    # 2 
        offsets1.append(Vector2(1, -1))     # 3
        
        # [ ] [ ] [ ]
        # [ ] [ ] [0]
        # [ ] [ ] [ ]
        offsets2 = list()
        offsets1.append(Vector2(1, 0))      # 0
        
        cls.apply_pass(grid1 , offsets1 ,offsets2 ,False)
        cls.apply_pass(grid2 , offsets1 ,offsets2 ,False)

        # Pass 2

        # [ ] [ ] [ ]
        # [ ] [ ] [0]
        # [2] [1] [3]
        offsets1.clear()
        offsets1.append(Vector2(1, 0))  # 0
        offsets1.append(Vector2(0, 1))  # 1
        offsets1.append(Vector2(-1, 1)) # 2
        offsets1.append(Vector2(1, 1))  # 3

        # [ ] [ ] [ ]
        # [0] [ ] [ ]
        # [ ] [ ] [ ]
        offsets2.clear()
        offsets2.append(Vector2(-1, 0)) # 0


        # make Img data
        out_img = np.zeros((width,height),dtype=np.float32)
        for y in range(height):
            for x in range(width):
                distance1 = grid1.get_dist(x, y)
                distance2 = grid2.get_dist(x, y)
                distance = distance1.length() - distance2.length()
                distance = (1 + max(-1, min(distance * scale, 1))) / 2.0
                out_img[x][y] = distance

        cv2.imshow("SDF Image", out_img.astype(np.uint8))
        
        #cv2.imwrite(p_output_image_path,out_img.astype(np.uint8))
        