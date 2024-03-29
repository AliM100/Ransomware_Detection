import pandas as pd
import os
import math
from keras.preprocessing.image import ImageDataGenerator
import shutil


class prepare_data:
  def __init__(self,data_path,img_path,class_index):
    self.data_path=data_path
    self.img_path=img_path
    self.class_index=class_index

  def combine_datasets(self,malevis_data_path):
    for folders in os.listdir(malevis_data_path+'/train'):
      if folders in self.class_index:
        shutil.move(malevis_data_path+'/train/'+folders,self.img_path)
        #Incase you need also calidation images
        # for file in os.listdir(malevis_data_path+"/val/"+folders):
        #   shutil.move(malevis_data_path+"/val/"+folders+'/'+file,self.img_path+'/'+folders)

  def prepare_malvis(self,malvis_path):
    for folder_name in os.listdir(os.path.join(malvis_path, 'train')):
        train_folder_path = os.path.join(malvis_path, 'train', folder_name)
        # val_folder_path = os.path.join(malvis_path, 'val', folder_name)

        # Check if both train and val folders exist for the same folder_name
        if os.path.exists(train_folder_path): #and os.path.exists(val_folder_path):
            # Define the destination path in the merged_data directory
            destination_path = os.path.join(self.img_path, folder_name)

            # Create the destination path
            os.makedirs(destination_path, exist_ok=True)

            # Copy the contents of train_folder to destination_path
            for item in os.listdir(train_folder_path):
                shutil.copy(os.path.join(train_folder_path, item), destination_path)

            # Append the contents of val_folder to destination_path
            # for item in os.listdir(val_folder_path):
            #     shutil.copy(os.path.join(val_folder_path, item), destination_path)    
    
  def split_train_test_val(self,df_train, df_test, df_val, classe,split_ratio_test,split_ratio_val):

    images = os.listdir(os.path.join(self.img_path,classe))
    images = [classe+"/"+elem for elem in images]
    index = self.class_index[classe]

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

  
  def create_csv_data(self):

    df_train_complete = pd.DataFrame(columns = ['img_code','target'])
    df_test_complete = pd.DataFrame(columns = ['img_code','target'])
    df_val_complete = pd.DataFrame(columns = ['img_code','target'])

    for classe in self.class_index.keys():
      df_train = pd.DataFrame(columns = ['img_code','target'])
      df_test = pd.DataFrame(columns = ['img_code','target'])
      df_val = pd.DataFrame(columns = ['img_code','target'])

      df_train, df_test, df_val = self.split_train_test_val(df_train, df_test, df_val,classe, 0.2,0.2)
      df_train_complete = pd.concat([df_train_complete, df_train], ignore_index = True)
      df_test_complete = pd.concat([df_test_complete, df_test], ignore_index = True)
      df_val_complete =pd.concat([df_val_complete, df_val], ignore_index = True)



    df_train_complete.to_csv(os.path.join(self.data_path+"/csvs/train.csv"))
    df_test_complete.to_csv(os.path.join(self.data_path+"/csvs/test.csv"))
    df_val_complete.to_csv(os.path.join(self.data_path+"/csvs/val.csv"))



def sample_rows_per_class(group,desired_samples_per_class):
      return group.sample(min(desired_samples_per_class, len(group)), replace=False)

class load_data:

  def __init__(self,img_path,data_csvs,target_size_custom,batch_size) :
    self.img_path=img_path
    self.data_csvs=data_csvs
    self.target_size_custom=target_size_custom
    self.batch_size=batch_size
    
  
  def train_data(self):
    target_column = 'target'
    train_df_partial=pd.read_csv(os.path.join(self.data_csvs,"train.csv"))
    val_df=pd.read_csv(os.path.join(self.data_csvs,"val.csv"))

    train_balanced_df = train_df_partial
    val_balanced_df = val_df
    # desired_samples_per_class= train_df_partial[target_column].value_counts().mean() + (10 * float(train_df_partial[target_column].value_counts().mean())) / 100.0
    # train_balanced_df = train_df_partial.groupby(target_column, group_keys=False, sort=False).apply(lambda x: sample_rows_per_class(x, int(desired_samples_per_class)))
    # train_balanced_df = train_balanced_df.sample(frac=1).reset_index(drop=True)


    # desired_samples_per_class= val_df[target_column].value_counts().mean() + (10 * float(val_df[target_column].value_counts().mean())) / 100.0
    # val_balanced_df = val_df.groupby(target_column, group_keys=False, sort=False).apply(lambda x: sample_rows_per_class(x, int(desired_samples_per_class)))
    
    # val_balanced_df = val_balanced_df.sample(frac=1).reset_index(drop=True)

    datagen = ImageDataGenerator(
          rescale=1 / 255.0)
    
    
    # train_balanced_df =pd.concat([train_balanced_df, val_balanced_df], ignore_index = True)
    
    train = datagen.flow_from_dataframe(
      dataframe=train_balanced_df,
      directory=self.img_path,
      x_col="img_code",
      y_col="target",
      target_size=self.target_size_custom,
      # color_mode = "grayscale",
      batch_size=self.batch_size,
      class_mode="sparse",
      shuffle=True,
      seed=42
    )

    val = datagen.flow_from_dataframe(
      dataframe=val_balanced_df,
      directory=self.img_path,
      x_col="img_code",
      y_col="target",
      target_size=self.target_size_custom,
      # color_mode = "grayscale",
      batch_size=self.batch_size,
      class_mode="sparse",
      shuffle=True,
      seed=42
    )
    return train,val



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
      # color_mode = "grayscale",
      batch_size=self.batch_size,
      class_mode="sparse",
      shuffle=False,
      seed=42
  )

    return test_gen

