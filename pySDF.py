# _*_ coding=utf-8 _*_
# MineClever 's 2D SDF Generator~

import os, sys
import cv2 as cv2
import numpy as np
import math as math

# def type
class Vector2 ():
    _debug = True
    def __init__(self, x=0,y=0):
        self.data = np.array((x,y),dtype=float)

    @property
    def x (self):
        return self.data[0]

    @x.setter
    def x (self, value):
        self.data[0] = (value)
        
    @property
    def y (self):
        return self.data[1]

    @y.setter
    def y (self, value):
        self.data[1] = (value)

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
    _debug = False

    class Grid ():
        
        def __init__(self, width:int , height:int):  
            self.width = width
            self.height = height
            self.size = Vector2(self.width,self.height)
            self.distances = [Vector2(0,0)]* (self.width*self.height)

        def __str__ (self):
            return "width:{},height:{}".format(self.size.x, self.size.y)
        
        def has (self, x:int, y:int) -> bool:
            return (0 <= x and x < self.size.x and 0 <= self.size.y and y < self.size.y)

        def _index (self, x:int, y:int) -> int:
            # print ("x : {}, y: {}".format(x,y))
            return int(y * self.size.x + x)

        
        def get_size(self) -> Vector2:
            return self.size

        def get_dist(self, x:int, y:int) -> Vector2:
            # print (self._index(x,y))
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
        size = len(p_offsets)
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
                # else:
                #     x = 0
                while (x < width):
                    cls.apply_offsets(p_grid, x, y, p_offsets2)
                    x += 1
                y -= 1
        else:
            y = 0
            x = 0
            while (y < height) :
                while (x < width):
                    cls.apply_offsets(p_grid, x, y, p_offsets1)
                    x += 1
                else:
                    x = width - 1
                while (x > 0):
                    cls.apply_offsets(p_grid, x, y, p_offsets2)
                    x -= 1
                y += 1

        

    @staticmethod
    def _bind_methods():
        pass

    @classmethod
    def do_sdf (cls, p_input_image_path='', p_img_size = 512):
        # read img by openCV
        img = cv2.imread(p_input_image_path,cv2.IMREAD_UNCHANGED)

        width = img.shape[0]
        height = img.shape[1]
        max_len = height if height > width else width
        print (img.shape)
        
        if max_len > p_img_size:
            print("do scale")
            scale_fac = p_img_size / max_len
            print (scale_fac)
            img = cv2.resize(img,dsize=(int(width*scale_fac) , int(height* scale_fac)),interpolation=cv2.INTER_LANCZOS4)
            width = img.shape[0]
            height = img.shape[1]
        
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print (img.shape)
        
        # Initialise grids
        grid1 = cls.Grid(width,height)
        grid2 = cls.Grid(width,height)
        DISTANT = 999999

        for y in range(height):
            for x in range(width):
                img_pixel = img[x][y][2] / 255 # convert 255 -> 1.0
                distance = 0 if img_pixel > 0.5 else DISTANT
                grid1.set_dist(x, y, Vector2(distance, distance))
                substract_dist = DISTANT - distance
                grid2.set_dist(x, y, Vector2(substract_dist, substract_dist))


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

        cls.apply_pass(grid1 , offsets1 ,offsets2 ,True)
        cls.apply_pass(grid2 , offsets1 ,offsets2 ,True)
        
        # make Img data
        out_data_array = np.zeros((width,height),dtype=np.float16)
        # print(out_img.shape)
        for y in range(height):
            for x in range(width):
                distance1 = grid1.get_dist(x, y)
                distance2 = grid2.get_dist(x, y)
                distance = distance2.length() - distance1.length()
                out_data_array[x][y] = distance
        return out_data_array

    @classmethod
    def do_genshin_sdf_sequence_export (cls, p_input_image_path='',p_output_image_path='', p_scale = 1.25, p_img_size = 512):
        img_data_array = cls.do_sdf(p_input_image_path, p_img_size)
        max_val = np.max(img_data_array)
        for y in range(p_img_size):
            for x in range(p_img_size):
                distance = img_data_array[x][y]
                #img_data_array[x][y] = (1 + max(-1, min(distance * p_scale, 1))) / 2.0
                img_data_array[x][y] = distance / max_val * p_scale
        out_img_scaled = np.clip(img_data_array *255,0,255).astype('uint8')
        cv2.imwrite(p_output_image_path,out_img_scaled)


    @classmethod
    def do_genshin_sdf_blend_export (cls, p_input_image_path_list=[''],p_output_image_path='', p_scale = 1.25, p_img_size = 512):
        if not p_input_image_path_list:
            return
        
        def lerp(a, b, value):
            # type: (float, float, float) -> float
            return a + value * (b - a)

        def saturate(a):
            # type: (float) -> float
            return min(1,max(0,a))

        def smoothstep( a,  b,  x):
            # type: (float, float, float) -> float
            t = saturate((x - a)/(b - a))
            return t*t*(3.0 - (2.0*t))
        
        img_counts = p_input_image_path_list.__len__()
        # NOTE: process all images, last img is export img
        all_img_data_array = np.zeros((img_counts+1, p_img_size, p_img_size),dtype=np.float16)
        
        for index in range(img_counts):
            img_path = p_input_image_path_list[index]
            img_data_array = cls.do_sdf(img_path, p_img_size)
            max_val = np.max(img_data_array)
            for y in range(p_img_size):
                for x in range(p_img_size):
                    distance = img_data_array[x][y]
                    # NOTE: normalize && scale
                    all_img_data_array[index][x][y] = np.clip(distance / max_val * p_scale, 0, 1) 
                    # img_data_array[x][y] = np.clip(distance / max_val, 0, 1) * p_scale
                    
        # Blend Img
        blend_delta = 0.01
        for j in range(1, 256+1):
            blend_rank = j / 256
            for i in range(img_counts):
                if (i / img_counts < blend_rank) and (blend_rank <= (i+1) / img_counts ):
                    for y in range(p_img_size):
                        for x in range(p_img_size):
                            mix_value = lerp(all_img_data_array[i][x][y], all_img_data_array[i+1][x][y], blend_rank*8-i)
                            smooth_bank= smoothstep(0.5 - blend_delta, 0.5 + blend_delta, mix_value)
                            cur_img_col = all_img_data_array[img_counts][x][y]
                            all_img_data_array[img_counts][x][y] = ((j-1) * cur_img_col + smooth_bank)/j
                            break

        out_img_scaled = np.clip(all_img_data_array[img_counts] *255,0,255).astype('uint8')
        cv2.imwrite(p_output_image_path,out_img_scaled)



    @classmethod
    def do_genshin_sdf_blend_export_method2 (cls, p_input_image_path_list=[''],p_output_image_path='', p_scale = 1.25, p_img_size = 512):
        if not p_input_image_path_list:
            return
        
        def lerp(a, b, value):
            # type: (float, float, float) -> float
            return a + value * (b - a)

        def saturate(a):
            # type: (float) -> float
            return min(1,max(0,a))

        def smoothstep( a,  b,  x):
            # type: (float, float, float) -> float
            t = saturate((x - a)/(b - a))
            return t*t*(3.0 - (2.0*t))
        
        img_counts = p_input_image_path_list.__len__()
        # NOTE: process all images, last img is export img
        all_img_data_array = []
        
        for index in range(img_counts):
            img_data = np.zeros((p_img_size, p_img_size),dtype=np.float16)
            img_path = p_input_image_path_list[index]
            img_data_array = cls.do_sdf(img_path, p_img_size)
            max_val = np.max(img_data_array)
            for y in range(p_img_size):
                for x in range(p_img_size):
                    distance = img_data_array[x][y]
                    # NOTE: normalize && scale
                    img_data[x][y] = np.clip(distance / max_val * p_scale, 0, 1)
            else:
                all_img_data_array.append(img_data)                    
        all_img_data_array.append(np.zeros((p_img_size, p_img_size),dtype=np.float16)) 

        # Blend Img
        blend_delta = 0.01
        delta_cut = 16

        for i in range(img_counts):
            img_data = all_img_data_array[i]
            if cls._debug :
                print("Current Index : {}".format(i))
                out_img_scaled = np.clip(img_data *255,0,255).astype('uint8')
                mixed_path = os.path.splitext(p_output_image_path)
                cv2.imwrite(mixed_path[0]+str(i)+mixed_path[1],out_img_scaled)
            next_index = i+1
            if next_index  == img_counts:
                continue
            next_img_data = all_img_data_array[next_index]
            temp_img_data = np.zeros((p_img_size, p_img_size),dtype=np.float16)
            for i in range(delta_cut+1):
                sdf_lerp_val = i/delta_cut
                for y in range(p_img_size):
                    for x in range(p_img_size):
                        cur_img_distance = img_data[x][y]
                        next_img_distance = next_img_data[x][y]
                        sample_val = lerp(cur_img_distance, next_img_distance, sdf_lerp_val)
                        smooth_val = smoothstep(0.5 - blend_delta, 0.5 +blend_delta ,sample_val)
                        temp_img_data[x][y] += smooth_val
            else:
                temp_img_data /= delta_cut
                all_img_data_array[img_counts] += temp_img_data
        else:
            # Note : get final value
            all_img_data_array[img_counts] /= img_counts
                        
        out_img_scaled = np.clip(all_img_data_array[img_counts] *255,0,255).astype('uint8')
        cv2.imwrite(p_output_image_path,out_img_scaled)