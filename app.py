
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model



st.title("MUltiple Fruits Images Classification")

#setting the image_size 
img_width = 180
img_height = 180

data_cat=['apple',
          'banana',
          'beetroot',
          'bell pepper',
          'cabbage',
          'capsicum',
          'carrot',
          'cauliflower',
          'chilli pepper',
          'corn',
          'cucumber',
          'eggplant',
          'garlic',
          'ginger',
          'grapes',
          'jalepeno',
          'kiwi',
          'lemon',
          'lettuce',
          'mango',
          'onion',
          'orange',
          'paprika',
          'pear',
          'peas',
          'pineapple',
          'pomegranate',
          'potato',
          'raddish',
          'soy beans',
          'spinach',
          'sweetcorn',
          'sweetpotato',
          'tomato',
          'turnip',
          'watermelon']

# Loading the model
model = tf.keras.models.load_model('Image_classify0.keras')

image_name = st.text_input("Enter Image_name:")
if st.button("Predict"):
    image_load= tf.keras.utils.load_img(image_name, target_size=(img_height,img_width))
    img_arr = tf.keras.utils.array_to_img(image_load)
    img_bat=tf.expand_dims(img_arr,0)

    predict = model.predict(img_bat)
    # st.image()

    score = tf.nn.softmax(predict)
    st.write('Veg/Fruit in image is ' +str(data_cat[np.argmax(score)])+'with accuracy of'+ np.max(score)*100)