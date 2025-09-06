# MobileNetV2-Animal-Classifier
A deep learning model that classifies 90 different animal species using TensorFlow and MobileNetV2.

# Animal Image Classifier

## Project Overview
This project demonstrates a multi-class image classifier trained to identify 90 different animal species. The model is built using TensorFlow and Keras, leveraging the power of **transfer learning** with a pre-trained **MobileNetV2** model. The entire project is optimized to run efficiently on Google Colab.

## Dataset
The dataset used contains a total of **5400 images**, divided among 90 unique animal classes.
- **Training Set**: 4320 images (80%) used to train the model.
- **Validation Set**: 1080 images (20%) used to evaluate model performance.

## Methodology
The core of this project relies on a few key techniques to achieve high performance and fast training:
- **Transfer Learning**: We use MobileNetV2, a model pre-trained on a massive dataset, as a feature extractor. This allows our model to learn efficiently with a smaller dataset.
- **Data Pipeline Optimization**: The dataset is stored in a single compressed file (`.zip`) and unzipped directly to the high-speed local disk of the Colab environment, eliminating I/O bottlenecks from Google Drive.
- **Mixed-Precision Training**: To significantly reduce training time, we use a mix of 16-bit and 32-bit floating-point types, which leverages the specialized hardware on GPUs like the T4.

## How to Run the Project
1.  **Open Google Colab**: Go to `colab.research.google.com`.
2.  **Create a New Notebook**: Go to `File > New notebook`.
3.  **Upload the Code**: Copy the entire code from the `animal_classifier.ipynb` file in this repository and paste it into a cell in your Colab notebook.
4.  **Upload the Dataset**: Upload your `animals.zip` file to your Google Drive. Make sure to update the `ZIP_FILE_PATH` variable in the notebook to match its location.
5.  **Run the Cells**: Execute the notebook cells sequentially. The code will handle everything from data loading and training to making predictions.

## Results
After 10 epochs of training, the model achieves the following performance on the validation set:
- **Validation Loss**: [Paste your final validation loss here]
- **Validation Accuracy**: [Paste your final validation accuracy here]

### Example Prediction
Here is an example of the model predicting a **goose** with high confidence.
