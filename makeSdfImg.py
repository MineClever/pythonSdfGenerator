# _*_ coding=utf-8 _*_
import os
from  pySDF import *


img_dir_path = os.path.split(os.path.abspath(__file__))[0]
print(img_dir_path)

files_list = list()
for root,dir,files in os.walk(img_dir_path):
    for file in files:
        if file.endswith("png"):
            print("using %s" % (file))
            file_full_path = os.path.join(root, file)
            files_list.append(file_full_path)

print("File input count : %d" %(len(files_list)))

export_dir = os.path.join(img_dir_path,"SDF")
if not os.path.exists(export_dir):
    os.mkdir(export_dir)
    
for file in files_list:
    export_short_name = os.path.splitext(os.path.split(file)[1])[0] + "_sdf.png"
    export_full_path = os.path.join(export_dir,export_short_name).replace("\\","/")
    SSEDT8.do_genshin_sdf_sequence_export(file, export_full_path,p_scale=1.25, p_img_size= 128)
    print("Finish export : %s" % (export_short_name))

input()