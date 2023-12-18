import pandas as pd
import os
import math
from keras.preprocessing.image import ImageDataGenerator


class_index = {'Adialer.C': 0,
                'Agent.FYI': 1,
                'Allaple.A': 2,
                'Allaple.L': 3,
                'Alueron.gen!J': 4,
                'Autorun.K': 5,
                'C2LOP.P': 6,
                'C2LOP.gen!g': 7,
                'Dialplatform.B': 8,
                'Dontovo.A': 9,
                'Fakerean': 10,
                'Instantaccess': 11,
                'Lolyda.AA1': 12,
                'Lolyda.AA2': 13,
                'Lolyda.AA3': 14,
                'Lolyda.AT': 15,
                'Malex.gen!J': 16,
                'Obfuscator.AD': 17,
                'Rbot!gen': 18,
                'Skintrim.N': 19,
                'Swizzor.gen!E': 20,
                'Swizzor.gen!I': 21,
                'VB.AT': 22,
                'Wintrim.BX': 23,
                'Yuner.A': 24,
                'Benign':25}



def split_train_test_val(img_path,df_train, df_test, df_val, classe,split_ratio_test,split_ratio_val):

  images = os.listdir(os.path.join(img_path,classe))
  images = ["./"+classe+"/"+elem for elem in images]
  index = class_index[classe]

  dim = len(images)
  size_test = math.floor(dim*split_ratio_test)
  size_val = math.floor((dim) * split_ratio_val)

  df_test['img_code'] = images[:size_test]
  df_val['img_code'] = images[size_test:size_test+size_val]
  df_train['img_code'] = images[size_test+size_val:]

  df_test['target'] = classe
  df_val['target'] =classe
  df_train['target'] = classe
  print(classe, df_train.shape, df_test.shape, dim)

  return df_train, df_test, df_val

def create_csv_data(data_path,img_path):

  df_train_complete = pd.DataFrame(columns = ['img_code','target'])
  df_test_complete = pd.DataFrame(columns = ['img_code','target'])
  df_val_complete = pd.DataFrame(columns = ['img_code','target'])

  for classe in class_index.keys():
    df_train = pd.DataFrame(columns = ['img_code','target'])
    df_test = pd.DataFrame(columns = ['img_code','target'])
    df_val = pd.DataFrame(columns = ['img_code','target'])

    df_train, df_test, df_val = split_train_test_val(img_path,df_train, df_test, df_val,classe, 0.2,0.2)
    df_train_complete = pd.concat([df_train_complete, df_train], ignore_index = True)
    df_test_complete = pd.concat([df_test_complete, df_test], ignore_index = True)
    df_val_complete =pd.concat([df_val_complete, df_val], ignore_index = True)



  df_train_complete.to_csv(os.path.join(data_path+"/csvs/train.csv"))
  df_test_complete.to_csv(os.path.join(data_path+"/csvs/test.csv"))
  df_val_complete.to_csv(os.path.join(data_path+"/csvs/val.csv"))

class load_data:

  def __init__(self,img_path,data_csvs,target_size_custom,batch_size) :
    self.img_path=img_path
    self.data_csvs=data_csvs
    self.target_size_custom=target_size_custom
    self.batch_size=batch_size


  def train_data(self):
    train_df_partial=pd.read_csv(os.path.join(self.data_csvs,"train.csv"))
    val_df=pd.read_csv(os.path.join(self.data_csvs,"val.csv"))

    datagen = ImageDataGenerator(
          rescale=1 / 255.0)
    
    train_df =pd.concat([train_df_partial, val_df], ignore_index = True)
    print("train_df",train_df)
    train_gen = datagen.flow_from_dataframe(
      dataframe=train_df,
      directory=self.img_path,
      x_col="img_code",
      y_col="target",
      target_size=self.target_size_custom,
      #color_mode = "grayscale",
      batch_size=self.batch_size,
      class_mode="categorical",
      shuffle=True,
      seed=42
    )

    val_gen = datagen.flow_from_dataframe(
      dataframe=val_df,
      directory=self.img_path,
      x_col="img_code",
      y_col="target",
      target_size=self.target_size_custom,
      #color_mode = "grayscale",
      batch_size=self.batch_size,
      class_mode="categorical",
      shuffle=True,
      seed=42
    )
    return train_gen,val_gen



  def test_data(self):
    
    test_df=pd.read_csv(os.path.join(self.data_csvs,"test.csv"))

    datagen = ImageDataGenerator(
          rescale=1 / 255.0)
    
    test_gen = datagen.flow_from_dataframe(
      dataframe=test_df,
      directory=self.img_path,
      x_col="img_code",
      y_col="target",
      target_size=self.target_size_custom,
      #color_mode = "grayscale",
      batch_size=self.batch_size,
      class_mode="categorical",
      shuffle=False,
      seed=42
  )

    return test_gen

