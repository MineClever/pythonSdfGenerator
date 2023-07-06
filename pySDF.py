# _*_ coding=utf-8 _*_
# MineClever 's 2D SDF Generator~

import os, sys
import cv2 as cv2
import numpy as np
import math as math

# def math
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
    def do_sdf(cls, p_input_image_path='', p_img_size=512):
        # read img by openCV
        print("Start process image : {}".format(p_input_image_path))
        img = cv2.imread(p_input_image_path, cv2.IMREAD_UNCHANGED)
        width = img.shape[0]
        height = img.shape[1]
        max_len = height if height > width else width
        print("Origin Image size: ", img.shape)

        if max_len > p_img_size:
            scale_fac = p_img_size / max_len
            print("Do scale fac :", scale_fac)
            img = cv2.resize(img,
                             dsize=(int(width * scale_fac),
                                    int(height * scale_fac)),
                             interpolation=cv2.INTER_LINEAR)
            width = img.shape[0]
            height = img.shape[1]

        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("SDF Image Size: ", img.shape)

        # Initialise grids
        grid1 = cls.Grid(width, height)
        grid2 = cls.Grid(width, height)
        DISTANT = 999999

        for y in range(height):
            for x in range(width):
                img_pixel = img[x][y][2] / 255  # convert 255 -> 1.0
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
        offsets1.append(Vector2(-1, 0))  # 0
        offsets1.append(Vector2(0, -1))  # 1
        offsets1.append(Vector2(-1, -1))  # 2
        offsets1.append(Vector2(1, -1))  # 3

        # [ ] [ ] [ ]
        # [ ] [ ] [0]
        # [ ] [ ] [ ]
        offsets2 = list()
        offsets1.append(Vector2(1, 0))  # 0

        cls.apply_pass(grid1, offsets1, offsets2, False)
        cls.apply_pass(grid2, offsets1, offsets2, False)

        # Pass 2

        # [ ] [ ] [ ]
        # [ ] [ ] [0]
        # [2] [1] [3]
        offsets1.clear()
        offsets1.append(Vector2(1, 0))  # 0
        offsets1.append(Vector2(0, 1))  # 1
        offsets1.append(Vector2(-1, 1))  # 2
        offsets1.append(Vector2(1, 1))  # 3

        # [ ] [ ] [ ]
        # [0] [ ] [ ]
        # [ ] [ ] [ ]
        offsets2.clear()
        offsets2.append(Vector2(-1, 0))  # 0

        cls.apply_pass(grid1, offsets1, offsets2, True)
        cls.apply_pass(grid2, offsets1, offsets2, True)

        # make Img data
        out_data_array = np.zeros((width, height), dtype=np.float32)
        # print(out_img.shape)
        for y in range(height):
            for x in range(width):
                distance1 = grid1.get_dist(x, y)
                distance2 = grid2.get_dist(x, y)
                distance = distance2.length() - distance1.length()
                out_data_array[x][y] = distance
        return out_data_array


class SSEDT8_Exporter(SSEDT8):
    _debug = True

    @classmethod
    def do_general_sdf_img_export (cls, p_input_image_path='',p_output_image_path='', p_scale = 1.25, p_img_size = 512):
        img_data_array = cls.do_sdf(p_input_image_path, p_img_size)
        for y in range(p_img_size):
            for x in range(p_img_size):
                distance = img_data_array[x][y]
                img_data_array[x][y] = (1 + max(-1, min(distance * p_scale, 1))) / 2.0
        data_max_value = (np.iinfo(np.uint16).max)
        out_img_scaled = np.clip(img_data_array *data_max_value, 0, data_max_value).astype(np.uint16)
        cv2.imwrite(p_output_image_path, out_img_scaled)

    @classmethod
    def do_genshin_sdf_img_export (cls, p_input_image_path='',p_output_image_path='', p_scale = 0.5, p_img_size = 512):
        img_data_array = cls.do_sdf(p_input_image_path, p_img_size)
        max_val = np.max(img_data_array)
        mid_scale = saturate(p_scale)
        for y in range(p_img_size):
            for x in range(p_img_size):
                distance = img_data_array[x][y]
                img_data_array[x][y] = np.clip(distance / (max_val * mid_scale) , 0, 1)
        data_max_value = (np.iinfo(np.uint16).max)
        out_img_scaled = np.clip(img_data_array *data_max_value, 0, data_max_value).astype(np.uint16)
        cv2.imwrite(p_output_image_path, out_img_scaled)

    @classmethod
    def do_genshin_sdf_mix_data(cls, p_img_a_path="", p_img_b_path=""):
        pass

    @classmethod
    def do_genshin_sdf_blend_export_method1(cls,
                                            p_input_image_path_list=[''],
                                            p_output_image_path='',
                                            p_scale=1.25,
                                            p_img_size=512,
                                            *args,
                                            **kw):
        if not p_input_image_path_list:
            return

        img_counts = p_input_image_path_list.__len__()
        # NOTE: process all images, last img is export img
        all_img_data_array = np.zeros((img_counts+1, p_img_size, p_img_size),dtype=np.float32)

        for index in range(img_counts):
            img_path = p_input_image_path_list[index]
            img_data_array = cls.do_sdf(img_path, p_img_size)
            max_val = np.max(img_data_array)
            for y in range(p_img_size):
                for x in range(p_img_size):
                    distance = img_data_array[x][y]
                    # NOTE: normalize && scale
                    all_img_data_array[index][x][y] = np.clip(distance / max_val * p_scale, 0, 1)

        # Blend Img
        blend_delta = 0.01
        for grey_val in range(1, 256):
            blend_rank = grey_val / 255
            for img_index in range(img_counts):
                if ((img_index / img_counts) < blend_rank) and (((img_index+1) / img_counts) >= blend_rank):
                    for y in range(p_img_size):
                        for x in range(p_img_size):
                            cur_img_data = all_img_data_array[img_index][x][y]
                            next_img_data = all_img_data_array[img_index + 1][x][y]
                            mix_value = lerp(cur_img_data, next_img_data, blend_rank * (img_counts) - img_index)
                            smooth_bank= smoothstep(0.5 - blend_delta, 0.5 + blend_delta, mix_value)
                            cur_img_val_data = all_img_data_array[img_counts][x][y]
                            cur_img_val_data = ((grey_val-1) * cur_img_val_data + smooth_bank)/grey_val
                else:
                    break

        out_img_scaled = np.clip(all_img_data_array[img_counts] *255,0,255).astype('uint8')
        cv2.imwrite(p_output_image_path,out_img_scaled)

    @classmethod
    def do_genshin_sdf_blend_export_method2(cls,
                                            p_input_image_path_list=[''],
                                            p_output_image_path='',
                                            p_scale=0.5,
                                            p_img_size=512,
                                            p_lerp_time=64,
                                            b_export_sdf = True,
                                            *args,
                                            **kw):
        if not p_input_image_path_list:
            return

        img_counts = p_input_image_path_list.__len__()
        # NOTE: process all images, last img is export img
        all_img_data_array = np.zeros([img_counts + 1, p_img_size, p_img_size], dtype=np.float32)
        mid_scale = saturate(p_scale)
        # TODO: usd multiProcess to generate sdf more fast!
        for index in range(img_counts):
            img_path = p_input_image_path_list[index]
            img_data_array = all_img_data_array[index]
            sdf_data_array = cls.do_sdf(img_path, p_img_size)
            max_val = np.max(sdf_data_array)

            for y in range(p_img_size):
                for x in range(p_img_size):
                    distance = sdf_data_array[x][y]
                    # NOTE: normalize && scale
                    # TODO: We should normalize line edge value as 0.5 (same as 128 of Grey 255)
                    # NOTE: 0.865 is magic number to fix value ...
                    img_data_array[x][y] = np.clip(distance / (max_val * mid_scale) , 0, 1)

        # NOTE: Blend Img
        print("Blending Mixed SDF Image ...")
        lerp_times = p_lerp_time # NOTE: 16 -> 64 times is good enough ...
        blend_delta = 1 / img_counts
        # TODO: Find average point value between two img , may get better linear interpolation ?
        # TODO: use multiProcess to Blend two image more fast!
        for cur_index in range(img_counts):
            print("Current Index : {}".format(cur_index))
            img_data = all_img_data_array[cur_index]
            if b_export_sdf:
                # NOTE: auto gen sdf file name
                mixed_path = os.path.splitext(p_output_image_path)
                out_img_path = mixed_path[0]+str(cur_index)+mixed_path[1]

                data_max_value = (np.iinfo(np.uint16).max)
                out_img_scaled = np.clip(img_data *data_max_value, 0, data_max_value).astype(np.uint16)
                cv2.imwrite(out_img_path, out_img_scaled)

            # NOTE: only mix img between two img
            next_index = cur_index+1
            if next_index  >= img_counts:
                continue

            next_img_data = all_img_data_array[next_index]
            temp_img_data = np.zeros((p_img_size, p_img_size),dtype=np.float32)
            for time in range(lerp_times+1):
                sdf_lerp_val = time / lerp_times
                for y in range(p_img_size):
                    for x in range(p_img_size):
                        cur_img_distance = img_data[x][y]
                        next_img_distance = next_img_data[x][y]
                        sample_val = lerp(cur_img_distance, next_img_distance, sdf_lerp_val)
                        smooth_val = smoothstep(0.0, 0.5 - blend_delta, sample_val)
                        temp_img_data[x][y] += smooth_val
            else:
                temp_img_data /= lerp_times
                all_img_data_array[img_counts] += temp_img_data
        else:
            # Note : get final value
            all_img_data_array[img_counts] /= img_counts
        data_max_value = (np.iinfo(np.uint16).max)
        out_img_scaled = np.clip(all_img_data_array[img_counts] *data_max_value,0,data_max_value).astype(np.uint16)
        cv2.imwrite(p_output_image_path,out_img_scaled)