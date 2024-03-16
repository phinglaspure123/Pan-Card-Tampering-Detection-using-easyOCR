# importing libraries
import pandas as pd
import numpy as np
import re
import cv2 
import streamlit as st
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import easyocr

plt.style.use('ggplot')


# Creating object of easyocr
reader = easyocr.Reader(['en','hi'], gpu = True)

# preprocessing the orienatation of the image 
def preprocess(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # contrast the image 
    image = cv2.convertScaleAbs(image, -1, alpha=0.9,beta=1.5) 

    # Check the image dimensions
    (h, w) = image.shape[:2]

    # If the image is in landscape orientation
    if w > h:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Rotate the image to portrait orientation
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 90, 1.0)
        image = cv2.warpAffine(image, M, (h, w), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        # Image is already in portrait orientation, grayscale of original image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return gray



def ocr_pan(image):
    results = reader.readtext(image)
    df=pd.DataFrame(results, columns=['bbox','text','conf'])
    df['text']=df['text'].apply(lambda x : re.sub(".*(नाम|तारीख ).","",x))
    pattern = re.compile(r'^[a-zA-Z0-9!@#$%^&*()_+-=|" "]+$')
    df['type']=df['text'].apply(lambda x:pattern.match(x))
    df['text']=df['text'].apply(lambda x :np.nan if bool(re.findall(r'(Digitally|sign|Physically|Valid|unless).*',x,flags=re.IGNORECASE)) else x) #removing sign cell
    df['text']=df['text'].apply(lambda x: np.nan if str(x).isnumeric() else x) #removing issue date since causing error
    global df2 
    df2=df.copy()
    df=df.dropna()
    df.drop(columns=['type'],inplace=True)
    df=df.reset_index(drop=True)
    return df

def plot_bounding_boxes(df, image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Add bounding boxes to the image
    for index, row in df.iterrows():
        # Get bounding box coordinates
        bbox = row['bbox']
        x1, y1 = bbox[0]
        x2, y2 = bbox[2]

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the image with bounding boxes using Streamlit
    st.image(image, channels="BGR", caption="Original Image with Bounding Boxes", use_column_width=True)


# show extracted data 
def ocr_text(df):
    try:
        st.subheader("Extracted Data:")
        new_rows=[]
        for i,r in df.iterrows():
            new_rows.append(r.to_dict())
        # Create a new DataFrame with the added rows
        df_new = pd.DataFrame(new_rows)
        
        keys=list(df_new['text'].iloc[::2])
        values=list(df_new['text'].iloc[1::2])

        df_new=pd.DataFrame({"key":keys,
                    "value":values})
        
        return df_new
    except:
        st.subheader("Raw Data")
        folder_path = "uploads"

        # Get the current working directory
        current_directory = os.getcwd()

        # List all files and directories in the uploads folder
        contents = os.listdir(folder_path)

        # Iterate through each item in the folder
        for item in contents:
            item_path = os.path.join(folder_path, item)
            # Check if the item is a file
            if os.path.isfile(item_path):
                # Extract the filename and full path
                file_name = os.path.basename(item_path)
                # Get the relative path
                relative_path = os.path.relpath(item_path, current_directory)
                # st.image(relative_path)
                
        results = reader.readtext(relative_path)
        df=pd.DataFrame(results, columns=['bbox','text','conf'])
        df['text']=df['text'].apply(lambda x : re.sub(".*(नाम|तारीख ).","",x))
        pattern = re.compile(r'^[a-zA-Z0-9!@#$%^&*()_+-=|" "]+$')
        df['type']=df['text'].apply(lambda x:pattern.match(x))
        df['text']=df['text'].apply(lambda x :np.nan if bool(re.findall(r'(Digitally|sign|Physically|Valid|unless).*',x,flags=re.IGNORECASE)) else x) #removing sign cell
        df['text']=df['text'].apply(lambda x: np.nan if str(x).isnumeric() else x) #removing issue date since causing error
        df=df.dropna()
        df.drop(columns=['type'],inplace=True)
        # removing sign photo if detected logic
        index_to_drop = df['conf'].idxmin()
        df.drop(index_to_drop, inplace=True)
        df=df.reset_index(drop=True)
        
        try:
            new_rows=[]
            for i,r in df.iterrows():
                new_rows.append(r.to_dict())
            # Create a new DataFrame with the added rows
            df_new = pd.DataFrame(new_rows)
            
            keys=list(df_new['text'].iloc[::2])
            values=list(df_new['text'].iloc[1::2])

            df_new=pd.DataFrame({"key":keys,
                        "value":values})
            
            return df_new
        except:
            return df