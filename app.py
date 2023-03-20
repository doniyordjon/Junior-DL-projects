import streamlit as st
import fastai
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp=pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


# title
st.title("Transportni klassifikasiya qiluvchi model")

#rasmni joylash
file = st.file_uploader('rasm yuklash', type=['png', 'jpeg', 'jpg', 'gif', 'svg'])

if file:
    st.image(file)

    #PIL convert
    img =PILImage.create(file)
    #model
    model = load_learner('transport_model.pkl')
    #prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f'Bashorat: {pred}')
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}%')
    
    #plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)