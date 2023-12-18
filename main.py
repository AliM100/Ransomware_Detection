import os
import sys
import os
from math import log
import numpy as np
import scipy as sp
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import dataset 
from dataset import load_data
from model import get_model
import tensorflow
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model



def train(dataloader,target_size_custom,save_checkpoints_path,batch_size):
    
    train_gen,val_gen=dataloader.train_data()
    with tensorflow.device('GPU'):
        model = get_model(target_size_custom)
        epochs = 10
        rlrp = ReduceLROnPlateau(monitor='loss', factor=0.2, verbose=1, patience=5, min_lr=0.000001) #

        history=model.fit(train_gen, validation_data=val_gen, batch_size=batch_size, epochs=epochs, callbacks=[rlrp])

        #saving model weights and history
        model.save(f'{save_checkpoints_path}/model.h5')
        
        hist_df = pd.DataFrame(history.history)

        with open(f"{save_checkpoints_path}/history.json", "w") as outfile:
            hist_df.to_json(outfile)




def test(dataloader,save_checkpoints_path):

    test_gen=dataloader.test_data()
   
    model=load_model(f"{save_checkpoints_path}/model.h5")
  
    
    pred=model.predict(test_gen)

    print(pred)



if __name__=="__main__":
    gpus = tensorflow.config.list_physical_devices('GPU')
    print(gpus)
    
    data_path="data"
    img_path="data/malimg_paper_dataset_imgs"
    data_csvs="data/csvs"
    save_checkpoints_path="data/checkpoint"
    batch_size=28
    os.makedirs(save_checkpoints_path,exist_ok=True)
    os.makedirs(data_csvs,exist_ok=True)
    
    if not os.path.exists(f"{data_csvs}/train.csv"):
        dataset.create_csv_data(data_path,img_path)
    
    target_size_custom = (256, 256)
    
    
    dataloader=load_data(img_path,data_csvs,target_size_custom,batch_size)

    train(dataloader,target_size_custom,save_checkpoints_path,batch_size)
    
    # test(dataloader,target_size_custom)