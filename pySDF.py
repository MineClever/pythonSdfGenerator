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

def clamp(value, min_val, max_val):
    # type: (float, float, float) -> float
    #return max(min(max_val, value), min_val)
    return np.clip(value ,min_val, max_val)


def saturate(a):
    # type: (float) -> float
    return clamp(a, 0, 1)

def smoothstep( a,  b,  x):
    # type: (float, float, float) -> float
    t = saturate((x - a)/(b - a))
    return t*t*(3.0 - (2.0*t))

# def type
class Vector2 ():
    _debug = True
    def __init__(self, x=0,y=0):
        self.data = np.array((x,y), dtype=np.float32)

    @property
    def x (self):
        # type: ()-> float
        return self.data[0]

    @x.setter
    def x (self, value):
        self.data[0] = (value)

    @property
    def y (self):
        # type: ()->np.float32
        return self.data[1]

    @y.setter
    def y (self, value):
        self.data[1] = (value)

    def __add__ (self, value):
        return Vector2((self.x+value.x),(self.y+value.y))

    def __mod__(self, b):
        # type: (Vector2)->np.float32
        return self.length() - b.length()

    def length_squared (self):
        # type: ()->np.float32
        return (self.x*self.x + self.y*self.y)

    def length (self):
        # type: ()->np.float32
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
            self.distances = np.array([Vector2(0,0)]* (self.width*self.height))

        def __str__ (self):
            return "width:{},height:{}".format(self.size.x, self.size.y)

        def has (self, x:int, y:int) -> bool:
            return (0 <= x and x < self.size.x and 0 <= self.size.y and y < self.size.y)

        def _index (self, x:int, y:int) -> int:
            return (y * self.size.x + x).astype(int)


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

    @staticmethod
    def read_img_data (p_input_image_path='', p_img_size=512, b_img_quad = False):
        # read img by openCV
        print("Start process image : {}".format(p_input_image_path))
        img = cv2.imread(p_input_image_path, cv2.IMREAD_UNCHANGED)
        width = img.shape[0]
        height = img.shape[1]
        print("Origin Image size: ", img.shape)
        if (width == p_img_size or height == p_img_size):
            return img

        # NOTE: calculate scale factor
        scale_fac_width = scale_fac_height =1
        if b_img_quad:
            scale_fac_width = p_img_size / width
            scale_fac_height = p_img_size / height
        else:
            max_len = height if height > width else width
            scale_fac_width = scale_fac_height = p_img_size / max_len

        # NOTE: scale now ...
        print("Do scale fac :", scale_fac_width, scale_fac_height)
        img = cv2.resize(img,
                            dsize=(int(width * scale_fac_width),
                                int(height * scale_fac_height)),
                            interpolation=cv2.INTER_LINEAR)
        return img

    @classmethod
    def do_sdf(cls, p_input_image_path='', p_img_size=512, b_img_quad=False):
        # read img by openCV
        img_data = cls.read_img_data(p_input_image_path, p_img_size, b_img_quad)
        if (img_data.shape.__len__()>=3):
            cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
            img_data = img_data[:,:,0]
        width = img_data.shape[0]
        height = img_data.shape[1]
        print("Process SDF Image Size: ", width, height)
        
        data_max_value = (np.iinfo(img_data.dtype).max)
        print("SDF Image bit depth as : {}, Max bit depth count as {}".format(img_data.dtype, data_max_value))
        img_data = img_data / data_max_value
        
        return cls._do_sdf(img_data, width, height)

    @classmethod
    def _do_sdf(cls, img_data, p_width, p_height, *args, **kw):
        # type: (cv2.Mat, int, int, ..., ...) -> np.ndarray

        grey_img_data = img_data
        width = p_width
        height = p_height
    
        # Initialise grids
        grid1 = cls.Grid(width, height)
        grid2 = cls.Grid(width, height)
        DISTANT = 999999

        # NOTE: Create as index map
        index_map_array = np.arange(width*height,dtype=np.uint32)

        def set_grids (img_index_array):
            x = (img_index_array // width)
            y = (img_index_array % width)
            img_pixel = grey_img_data[x][y]
            distance = 0 if (img_pixel > 0.5) else DISTANT
            grid1.set_dist(x, y, Vector2(distance, distance))
            substract_dist = DISTANT - distance
            grid2.set_dist(x, y, Vector2(substract_dist, substract_dist))

        vec_set_grids = np.vectorize(set_grids)
        vec_set_grids(index_map_array)

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
        # out_data_array = np.zeros((width, height), dtype=np.float32)

        def get_grids (img_index):
            x = (img_index // width)
            y = (img_index % width)
            distance1 = grid1.get_dist(x, y)
            distance2 = grid2.get_dist(x, y)
            return distance2 % distance1

        out_data_array = get_grids(index_map_array).reshape(width, height)
        return out_data_array


class SSEDT8_Exporter(SSEDT8):
    _debug = True

    @classmethod
    def do_general_sdf_img_export (cls, p_input_image_path='',p_output_image_path='', p_scale = 1.25, p_img_size = 512):
        sdf_data_array = cls.do_sdf(p_input_image_path, p_img_size, b_img_quad=True)
        def array_distance_process (distance):
            scaled_distance = distance * p_scale
            return (1 + np.clip(scaled_distance, -1, 1)) * 0.5

        img_data_array = array_distance_process(sdf_data_array)
        data_max_value = (np.iinfo(np.uint16).max)
        out_img_scaled = np.clip(img_data_array *data_max_value, 0, data_max_value).astype(np.uint16)
        cv2.imwrite(p_output_image_path, out_img_scaled)

    @classmethod
    def do_genshin_sdf_img_export (cls, p_input_image_path='',p_output_image_path='', p_scale = 0.5, p_img_size = 512):
        sdf_data_array = cls.do_sdf(p_input_image_path, p_img_size, b_img_quad=True)
        max_val = np.max(sdf_data_array)
        mid_scale = saturate(p_scale)
        def array_distance_process (distance):
            scaled_distance = distance / (max_val * mid_scale)
            return (1 + np.clip(scaled_distance, -1, 1)) * 0.5
        
        img_data_array = array_distance_process(sdf_data_array)
        data_max_value = (np.iinfo(np.uint16).max)
        out_img_scaled = np.clip(img_data_array *data_max_value, 0, data_max_value).astype(np.uint16)
        cv2.imwrite(p_output_image_path, out_img_scaled)

    @classmethod
    def do_genshin_sdf_mix_data(cls, p_img_a_path="", p_img_b_path="", p_output_image_path="", p_img_size=512, p_lerp_time=32, p_blend_delta = 0.01):

        # NOTE: Blend Img
        print("Blending Mixed SDF Image From {} and {}".format(p_img_a_path, p_img_b_path))
        lerp_times = p_lerp_time # NOTE: 16 -> 64 times is good enough ...
        blend_delta = p_blend_delta

        cur_img_data = cls.read_img_data(p_img_a_path)
        next_img_data = cls.read_img_data(p_img_b_path)

        def smooth_lerp_img_data(a_array, b_array, out_array, sdf_lerp_val):
            sample_val = lerp(a_array, b_array, sdf_lerp_val)
            smooth_val = smoothstep(0.5 - blend_delta, 0.5 + blend_delta, sample_val)
            out_array[0] += smooth_val

        temp_img_data = np.zeros((p_img_size, p_img_size),dtype=np.float32)

        lerp_val_1d_array = np.linspace(0, 1, num=lerp_times)
        for i in range(lerp_times):
            smooth_lerp_img_data(cur_img_data, next_img_data, [temp_img_data], lerp_val_1d_array[i])

        temp_img_data /= lerp_times

        data_max_value = (np.iinfo(np.uint16).max)
        print("Write Export map as {}, max bit depth count as {} ".format(p_output_image_path, data_max_value))
        out_img_scaled = np.clip(temp_img_data * data_max_value, 0, data_max_value).astype(np.uint16)
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

        for index in range(img_counts):
            print("\nCurrent Index : {}".format(index))
            img_path = p_input_image_path_list[index]
            sdf_data_array = cls.do_sdf(img_path, p_img_size, b_img_quad=True)
            max_val = np.max(sdf_data_array)

            def array_distance_process (distance):
                scaled_distance = distance / (max_val * mid_scale)
                return (1 + np.clip(scaled_distance, -1, 1)) * 0.5

            all_img_data_array[index] = array_distance_process(sdf_data_array)
            if b_export_sdf:
                cur_img_data = all_img_data_array[index]
                # NOTE: auto gen sdf file name
                mixed_path = os.path.splitext(p_output_image_path)
                out_img_path = mixed_path[0]+str(index)+mixed_path[1]

                data_max_value = (np.iinfo(np.uint16).max)
                out_img_scaled = np.clip(cur_img_data *data_max_value, 0, data_max_value).astype(np.uint16)
                print("Export SDF Img as {}".format(out_img_path))
                cv2.imwrite(out_img_path, out_img_scaled)


        # NOTE: Blend Img
        print("Blending Mixed SDF Image ...")
        lerp_times = p_lerp_time # NOTE: 16 -> 64 times is good enough ...
        blend_delta = clamp(1 / img_counts, 0.01, 0.05)
        def smooth_lerp_img_data (a_array, b_array, out_array, sdf_lerp_val):
            sample_val = lerp(a_array, b_array, sdf_lerp_val)
            smooth_val = smoothstep(0.5 - blend_delta, 0.5 + blend_delta, sample_val)
            out_array[0] += smooth_val

        lerp_val_1d_array = np.linspace(0, 1, num=lerp_times)

        # NOTE: create 3d array for img data array
        # def create_lerp_val_stack (lerp_var_array):
        #     # type: (np.ndarray) -> np.ndarray
        #     array_stack = np.zeros([lerp_times, p_img_size, p_img_size])
        #     for i in range(lerp_var_array.size()):
        #         val = lerp_var_array[i]
        #         val_layer = np.full([p_img_size, p_img_size], val)
        #         array_stack[i] = val_layer
        #     return array_stack
        # lerp_val_3d_array = create_lerp_val_stack(lerp_val_1d_array)

        for cur_index in range(img_counts):
            print("Current Index : {}".format(cur_index))
            cur_img_data = all_img_data_array[cur_index]

            # NOTE: only mix img between two img
            next_index = cur_index+1
            if next_index  >= img_counts:
                continue

            next_img_data = all_img_data_array[next_index]
            temp_img_data = np.zeros((p_img_size, p_img_size),dtype=np.float32)

            for i in range(lerp_times):
                smooth_lerp_img_data(cur_img_data, next_img_data, [temp_img_data], lerp_val_1d_array[i])

            temp_img_data /= lerp_times
            all_img_data_array[img_counts] += temp_img_data
        else:
            # Note : get final value
            all_img_data_array[img_counts] /= img_counts

        data_max_value = (np.iinfo(np.uint16).max)
        print("Write Export map as {}, max bit depth count as {} ".format(p_output_image_path, data_max_value))
        out_img_scaled = np.clip(all_img_data_array[img_counts] *data_max_value,0,data_max_value).astype(np.uint16)
        cv2.imwrite(p_output_image_path,out_img_scaled)