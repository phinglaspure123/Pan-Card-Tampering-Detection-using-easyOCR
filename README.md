
# Pan Card Text Detection using OCR
The "Pan Card Text Detection using OCR" project aims to extract text information from Pan Card images using Optical Character Recognition (OCR) technology. The project provides a user-friendly interface where users can upload Pan Card images, and the application will process the image to extract relevant text data. This extracted data is then presented to the user in a structured format for easy readability and interpretation.


## Key Features

- Text Detection: The project utilizes the EasyOCR library for text detection in Pan Card images. EasyOCR is a Python package that provides a simple interface for performing OCR tasks with high accuracy.
- User Interface: A streamlined user interface is implemented using Streamlit, allowing users to easily upload Pan Card images and view the extracted text data.
- Preprocessing: Image preprocessing techniques are applied to ensure optimal text extraction. This includes contrast adjustment and orientation correction to handle images in both landscape and portrait orientations.
- Bounding Box Visualization: Bounding boxes are overlaid on the original image to visually indicate the detected text regions, enhancing the understanding of the OCR process.
- Data Extraction: The extracted text data is processed to filter out irrelevant information and present only the essential details from the Pan Card.
- Data Presentation: The extracted text data is presented to the user in a structured DataFrame format, facilitating easy interpretation and further analysis if required.

## Libraries Used:

- Streamlit: For building the user interface and handling file uploads.
- OpenCV (cv2): For image processing tasks such as grayscale conversion and rotation.
- Pandas: For data manipulation and structuring the extracted text data.
- Matplotlib: For visualization purposes, particularly for displaying bounding boxes.
- EasyOCR: For performing optical character recognition on the Pan Card images.
## Project Components:
- tamp_detection.py: This module contains functions for preprocessing images, performing OCR on Pan Card images, filtering out irrelevant text data, and visualizing bounding boxes.
- app.py: The main application script built using Streamlit. It handles user interactions, file uploads, and integrates the functionalities from tamp_detection.py to process uploaded Pan Card images and display the extracted text data.
## Installation

requirements.txt

run this command in terminal
```bash
  pip install -r requirements.txt
```
    
## Authors

- [@phinglaspure123](https://github.com/phinglaspure123)

