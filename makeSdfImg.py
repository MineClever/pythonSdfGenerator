# _*_ coding=utf-8 _*_
import os
from  pySDF import *

def run_script_gen_genshin_face(*args, **kw):
    img_dir_path = os.path.join((os.path.dirname(os.path.abspath(__file__))), "Face").replace("\\","/")
    print("input image dir",img_dir_path)

    img_dir_files_list = [
        os.path.join(img_dir_path, f).replace("\\","/") for f in os.listdir(img_dir_path)
        if not os.path.isdir(os.path.join(img_dir_path, f))
    ]
    files_list = []
    for file in img_dir_files_list:
        if file.endswith("png"):
            print("using file path {}".format(file))
            files_list.append(file)
    files_list.sort()


    print("Input Image count : %d" %(len(files_list)))

    export_dir = os.path.join(img_dir_path,"SDF")
    if not os.path.exists(export_dir):
        os.mkdir(export_dir)

    print("Generate Mixed SDF Image ...")
    blend_export_full_path = os.path.join(export_dir,"mixed_sdf").replace("\\","/") + ".png"
    SSEDT8_Exporter.do_genshin_sdf_blend_export_method2(files_list,
                                                        blend_export_full_path,
                                                        p_scale=0.5,
                                                        p_img_size=256,
                                                        lerp_time=32,
                                                        b_export_sdf=True)

    os.system('pause')
    exit()


if __name__ == "__main__":
    run_script_gen_genshin_face()