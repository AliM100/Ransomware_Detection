
import sys
import os
from math import log
import numpy as np
import scipy as sp
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

def pe2hex(file_path, output_file_path):
  print('Processing '+file_path)
  file = bytearray(open(file_path, 'rb').read())
  key = "\0"
  with open(output_file_path, 'w') as output:
      for count, byte in enumerate(file, 1):
          output.write(
              f'{byte ^ ord(key[(count - 1) % len(key)]):#0{4}x}' + (
                  '\n' if not count % 16 else ' '))
          

def hex2img(array, output_img_path):
    if array.shape[1]!=16: #If not hexadecimal
        assert(False)
    b=int((array.shape[0]*16)**(0.5))
    b=2**(int(log(b)/log(2))+1)
    a=int(array.shape[0]*16/b)
    print(a,b,array.shape)
    array=array[:a*b//16,:]
    array=np.reshape(array,(a,b))
    im = Image.fromarray(np.uint8(array))
    #out = im.transpose(Image.FLIP_LEFT_RIGHT )
    im.save(output_img_path, "PNG")
    return im

def convert_data(pe_data_path,bytes_data_path,img_data_path,csv_data_path):
    files= os.listdir(pe_data_path)

    for counter, name in enumerate(files):
        name_output = name.split(".")[0]
        print(name_output)
        pe2hex(os.path.join(pe_data_path,name), os.path.join(bytes_data_path,name_output+".bytes"))

    df_benign = pd.read_csv(csv_data_path, index_col=False)
    already_transformed = list(df_benign['img_code'])
    benign_class_index = 25

    files= os.listdir(bytes_data_path)

    for counter, name in enumerate(files):
        name_output = name.split(".")[0]

        output_image_path = os.path.join(img_data_path,name_output+".png")
        relative_image_path = os.path.join("./benign_data/benign_imgs", name_output+".png")

        if(relative_image_path in already_transformed):
         continue

        print('Processing '+output_image_path)

        f=open(os.path.join(bytes_data_path,name), 'r')

        array=[]
        for line in f:
            xx=line.replace("\n", "").split(" ")

            if(len(xx) != 16 or "" in xx):
              continue

            array.append([int(i,16) if i!='??' else 0 for i in xx])

        img = hex2img(np.array(array),output_image_path)
        del array

        df_benign.loc[len(df_benign.index)] = [relative_image_path, benign_class_index]

        f.close()

    df_benign.to_csv(csv_data_path, index=False)
