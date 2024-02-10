import numpy as np
import cv2
import streamlit as st
from PIL import Image



def colorizer(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    
    prototxt = r"D:\Colorizer-master\Colorizer\Models\models_colorization_deploy_v2.prototxt"
    model = r"D:\Colorizer-master\Colorizer\Models\colorization_release_v2.caffemodel"
    points = r"D:\Colorizer-master\Colorizer\Models\pts_in_hull.npy"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)
    
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
    
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
 
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    
    colorized = (255 * colorized).astype("uint8")
    
    return colorized

##########################################################################################################
    
st.write("""
          # Colorize your Black and white image
          """
          )

st.write("This is an app to turn Black & White image into Color image")
st.write("Made by Yash & Pravin")

file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    img = np.array(image)
    
    st.text("Your original image")
    st.image(image, use_column_width=True)
    
    st.text("Your colorized image")
    color = colorizer(img)
    
    st.image(color, use_column_width=True)
    
    print("done!")





