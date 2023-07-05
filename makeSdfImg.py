# _*_ coding=utf-8 _*_
import os
from  pySDF import *


img_dir_path = os.path.join((os.path.dirname(os.path.abspath(__file__))), "Face").replace("\\","/")
print("input image dir",img_dir_path)


img_dir_files_list = [
    os.path.join(img_dir_path, f) for f in os.listdir(img_dir_path)
    if not os.path.isdir(os.path.join(img_dir_path, f))
]
files_list = []
for file in img_dir_files_list:
    if file.endswith("png"):
        print("using %s" % (file))
        files_list.append(file.replace("\\","/"))
files_list.sort()


print("File input count : %d" %(len(files_list)))

export_dir = os.path.join(img_dir_path,"SDF")
if not os.path.exists(export_dir):
    os.mkdir(export_dir)

# for file in files_list:
#     export_short_name = os.path.splitext(os.path.split(file)[1])[0] + "_sdf.png"
#     export_full_path = os.path.join(export_dir,export_short_name).replace("\\","/")
#     SSEDT8.do_genshin_sdf_sequence_export(file, export_full_path, p_scale=1.25, p_img_size= 128)
#     print("Finish export : %s" % (export_short_name))

print("Gen blend one ...")
blend_export_full_path = os.path.join(export_dir,"mixed_sdf").replace("\\","/") + ".png"
SSEDT8.do_genshin_sdf_blend_export_method2(files_list, blend_export_full_path, p_mid_scale=0.5, p_img_size= 64, lerp_time=32)

input("Press any key ...")
exit()