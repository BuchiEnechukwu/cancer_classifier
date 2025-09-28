# Cancer Classification Predictor Project

## Introduction
With increasing number of deaths which cancer contribute to, one huge challenge has been that cancer is majorly detected at advanced staged. Therefore, this project aimed to 
enhance early detection of cancer. This project specifically is to build a predictor app for cancer detection.

## The Importance of this Project?
Cancer diagnosis is complex and often requires expert review of medical images. This project explores how AI can support clinicians by: providing fast, consistent predictions from medical images; reducing human error in diagnostic workflows; and demonstrating how AI can be packaged into a simple, user-friendly tool.

This project is not a medical device, but a proof-of-concept for AI in healthcare.

## Data Source
The data was obtained from kaggle (https://www.kaggle.com/datasets/mikeytracegod/lung-cancer-risk-dataset/data). This contained over 9GB data of scanned images of cancer, reason the datasets could not be uploaded on GitHub, since it allows for 50MB and less.

## Tools
The tools used in this project was python on VScode. The libraries are Streamlit (for web app framework), TensorFlow / Keras (for deep learning), NumPy, Pandas, PIL (for data handling & image processing) and Joblib (for label encoding)

## How It Works
	1.	Upload a medical image (JPG, PNG, TIFF, BMP, etc.).
	2.	The app preprocesses the image into the right format.
	3.	A TensorFlow deep learning model predicts the cancer type.
	4.	The result shows: predicted label (human-friendly name), confidence score and a preview of the uploaded image

## Demo screenshots
#### Home page
<img width="2868" height="1448" alt="image" src="https://github.com/user-attachments/assets/270e4f1b-041c-431e-8d7e-252e5e270b26" />

#### Classifier page
<img width="2868" height="1448" alt="image" src="https://github.com/user-attachments/assets/9e483d26-49ce-4d99-b083-8bb7565bab15" />

#### Sample prediction preview
<img width="2868" height="1448" alt="image" src="https://github.com/user-attachments/assets/b4068c03-8d19-4321-b8f1-d1ace4b6813d" />

## App Link
Please visit the link, https://cancerclassifier.streamlit.app/ for a demo. Note that radiology image is required for testing.

## Author
Buchi Enechukwu
