# MobileNetV2-Animal-Classifier

## Project Overview
This project demonstrates a multi-class image classifier trained to identify 90 different animal species. The model is built using TensorFlow and Keras, leveraging the power of **transfer learning** with a pre-trained **MobileNetV2** model. The entire project is optimized to run efficiently on Google Colab.

## Dataset
The dataset used contains a total of **5400 images**, divided among 90 unique animal classes.
- **Training Set**: 4320 images (80%) used to train the model.
- **Validation Set**: 1080 images (20%) used to evaluate model performance.

## Dataset Download

The dataset is not included in the repository due to its large size. Please download the `animals.zip` file from the link below and upload it to your own Google Drive.

**Download Link:** https://drive.google.com/file/d/1alKeBWbtYZKO_Jz7rTCgLh073WK6EhOZ/view?usp=sharing

## Methodology
The core of this project relies on a few key techniques to achieve high performance and fast training:
- **Transfer Learning**: We use MobileNetV2, a model pre-trained on a massive dataset, as a feature extractor. This allows our model to learn efficiently with a smaller dataset.
- **Data Pipeline Optimization**: The dataset is stored in a single compressed file (`.zip`) and unzipped directly to the high-speed local disk of the Colab environment, eliminating I/O bottlenecks from Google Drive.
- **Mixed-Precision Training**: To significantly reduce training time, we use a mix of 16-bit and 32-bit floating-point types, which leverages the specialized hardware on GPUs like the T4.

# How to Run the Project

You can run this project in two ways:
* **Option A:** Using Google Colab (with Google Drive & a ZIP file)
* **Option B:** Using a local environment like Jupyter Lab (with the dataset folder)

---

## 🔹 Option A: Run on Google Colab

1.  **Open Google Colab**: Navigate to [colab.research.google.com](https://colab.research.google.com).

2.  **Create a New Notebook**: Go to `File` > `New notebook`.

3.  **Upload the Code**: Copy the entire code from `animal_classifier.ipynb` into a Colab notebook cell.

4.  **Upload the Dataset**: Upload your `animals.zip` file to your Google Drive.

5.  **Update Variables**: Modify the following variables in the notebook to match your file paths.

    ```python
    zip_path = "/content/drive/MyDrive/Data_animals/animals.zip"
    local_unzip_path = "/content/animals_unzipped"
    ```

6.  **Enable GPU Runtime (Recommended)**:
    * Go to `Runtime` > `Change runtime type`.
    * Select **GPU (T4)** from the dropdown menu.

7.  **Run the Cells**: Execute the cells sequentially to train, evaluate, and make predictions.

---

## 🔹 Option B: Run Locally on Jupyter Lab

1.  **Install Dependencies** (Python 3.10 or below is recommended):

    ```bash
    pip install tensorflow matplotlib pillow requests
    ```
     **Note on TensorFlow & Python Versions:**

    -> This project works best with **Python 3.10 or below**, where TensorFlow has better GPU support. 

    -> In higher versions (3.11+), TensorFlow may default to the CPU, which is significantly slower. 
    
    -> For the best performance, use Python 3.10 locally or use Google Colab with a GPU.

2.  **Open Jupyter Lab**:

    ```bash
    jupyter lab
    ```

3.  **Set Dataset Path**: Instead of using a ZIP file, point directly to your unzipped dataset folder. Use `r""` to handle Windows paths correctly.

    ```python
    # Example for Windows
    DATA_DIR = r"C:\Users\YourName\Documents\animals_dataset"

    # Example for macOS/Linux
    DATA_DIR = "/home/yourname/documents/animals_dataset"
    ```

4.  **Run the Notebook**: Execute the cells sequentially. The model will load images from `DATA_DIR`, train, and allow predictions.

---

## Quick Environment Check

Before running, you can add this snippet to a top cell to verify your setup:

```python
import sys
import tensorflow as tf
print("Python version:", sys.version.split()[0])
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU detected:", gpus)
else:
    print("⚠️ No GPU detected. Training will run on CPU (slower).")
```

* **GPU found** → You’re good to go! 🚀
* **No GPU found** → Consider installing the correct CUDA/cuDNN drivers for your system or switch to the Colab method.

---

## **Recommended**

 > If you want to avoid the headache of managing Python versions, CUDA drivers, and local dependencies, **just use Option A (Google Colab with a T4 GPU)**. It’s free, faster for training, and much easier to set up!

## Results
After 20 epochs of training, the model achieves the following performance on the validation set:
- **Validation Loss**: 0.6365
- **Validation Accuracy**: 90%
### Example Prediction
#### **1. Prediction on a Local Dataset Image**
This example shows the model correctly identifying a **dog** from an image included in your local dataset.

#### **2. Prediction on an Image from a URL**
This example demonstrates the model's ability to predict on images from the web, showing it correctly identifies a **lion** from a given URL.
