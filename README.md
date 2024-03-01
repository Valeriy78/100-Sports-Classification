# Sports Image Classifier
## Overview
Sports Image Classifier is a web application for classifying sports images using PyTorch and Streamlit. This project was developed for educational purposes in the field of machine learning. The model for image classification is the EfficientNet-B1 neural network, trained on the public [data set](https://www.kaggle.com/datasets/gpiosenka/sports-classification). This model is able to classify images of 100 sports with an accuracy of about 98%. 
You can see the data set analisys and the model training in [Kaggle notebook](https://www.kaggle.com/code/valeriipasko/100-sports-classification-models-competition).
The application is deployed on [Streamlit Community Cloud](https://100-sports-classification-znsvlnkgcwv7hqo2tm5zej.streamlit.app).
## Usage
To run Sports Image Classifier locally, you can clone the repository on your local computer.  Then, you can install the required packages in your environment by running:

```
pip install -r requirements.txt
```

To start the application, run the following command:

```
streamlit run app.py
```

This command will launch the application in your default web browser. 
After launching the application, you can upload a custom sports image and click the "Get Prediction" button to get the sports prediction.
