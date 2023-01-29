from keras.applications import ResNet152
from keras.applications.resnet import preprocess_input
import cv2
import numpy
from tqdm import tqdm
import torch
import numpy as np
import os


model=ResNet152(weights='imagenet',include_top=False)




def extract_features(path,video_name): 
    video_cv=cv2.VideoCapture(path)
    n_frames=int(video_cv.get(cv2.CAP_PROP_FRAME_COUNT))
    video_per_feat=None

    for i in tqdm(range(n_frames)):
        success,frame=video_cv.read()
        if success:
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame=cv2.resize(frame,(224,224)).reshape(-1,224,224,3)
            frame=preprocess_input(frame)
            feature=model.predict(frame)
            feature=feature.flatten()
            if video_per_feat is None:
                video_per_feat=feature
            else:
                video_per_feat=np.vstack((video_per_feat,feature))


    print(video_name)  
    print(video_per_feat.shape)
    video_cv.release()




path=r"C:\Users\yansh\OneDrive\Documents\majorproject\video_files"

os.chdir(path)

for file in os.listdir():
    if file.endswith(".mp4"):
        file_path=f"{path}\{file}"
        extract_features(file_path,file)







        






