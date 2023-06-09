import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
import platform
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath=pathlib.PosixPath


#title
st.title('Transportni klassifikatsiya qiluvchi model')

# rasmni yuklash
file = st.file_uploader('Rasmni yuklash', type=['jpg', 'png', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)

    # PIL convert
    img = PILImage.create(file)
    # model
    model = load_learner('transport_model.pkl')

    # prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

# plotting
fig = px.bar(x=probs*100, y=model.dls.vocab)
st.plotly_chart(fig)
