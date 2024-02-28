import streamlit as st
from PIL import Image
from model.classification import single_image_predict, get_class_name


def add_sidebar():
    st.sidebar.image('static/ball.jpg')
    st.sidebar.header("Info about the application")
    st.sidebar.write("This project was developed for educational purposes in the field of machine learning.")
    st.sidebar.write("The model for image classification is the neural network EfficientNet-B1, which was trained on the public [data set](https://www.kaggle.com/datasets/gpiosenka/sports-classification).")
    st.sidebar.write("You can see the data set analisys and the model training in [kaggle notebook](https://www.kaggle.com/code/valeriipasko/100-sports-classification-models-competition).")


def show_image(file):
    image = Image.open(file)
    st.image(image)
    st.write("")


def predict(file):
    if st.button("Get Prediction"):
        st.write("")
        pred_label = single_image_predict(file)
        st.write("<span class='pred_label'>Predicted label:</span>", unsafe_allow_html=True)
        st.write("<h3 class='sport'>", pred_label, "</h3>", unsafe_allow_html=True )


def main():

    st.set_page_config(
        page_title="Sports Image Classifier",
        page_icon=":basketball:",
        layout="wide",
        initial_sidebar_state="expanded"
        )
    
    with open('static/style.css') as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    add_sidebar()

    with st.container():
        st.title("Sports Image Classifier")
        st.markdown("### This application is developed to classify sports images. You can upload an arbitrary sports image to get a sports prediction.")

    file = st.file_uploader("Upload an image", type=['png', 'jpg'] )

    if file is not None:
        col1, col2 = st.columns([3,1])
        with col1:
            show_image(file)
        with col2:
            predict(file)


if __name__ == '__main__':
    main()
