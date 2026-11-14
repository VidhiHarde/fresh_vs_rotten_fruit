# Fresh vs Rotten Fruit Classification

A deep learning quality-screening prototype. It looks at a photo of a fruit and reports **which fruit it is**, **whether it is fresh or rotten**, and **how confident the model is** in each. It is built with a fine-tuned MobileNetV2 (PyTorch) and a Streamlit web app.

**Live app:** https://freshvsrottenfruit.streamlit.app/

**Fruits covered:** apple, banana, mango, orange (each fresh or rotten)


## Project overview

Checking fruit by eye is slow and subjective, and a rotten fruit can slip through unnoticed. This project builds a simple vision system that gives a consistent first check from surface appearance and flags doubtful fruit for a manual look.

The project covers:

- Fine-tuning a small CNN (MobileNetV2) with crop, rotation and brightness augmentation
- A two-step idea (fruit type, then freshness) implemented as one network with two output heads
- Evaluation with accuracy, F1, per-fruit performance, confusion matrices, inference latency and FPS
- A local extension: our own photos of fresh and naturally overripe fruit on different backgrounds and lighting
- A Streamlit app with Upload, Camera.


## Dataset

The data has two parts. The full dataset is **not** stored in this repository because of its size.

This is the drive link for our dataset : https://drive.google.com/drive/folders/1_98NFG5QR_Ecy4ukxOxm6dFZuIR5Sg8r?usp=sharing

The drive contains :

A. 3 folders of dataset : 
  1. fresh_vs_rotten_dataset.zip : The **main** dataset of the model, this contains the maximum number of images through which the model has been trained. It       contains 3 folder (`train(70%)`, `test(15%)`, `val(15%)`) and inside each there are 8 class folders (`fresh_apple`, `fresh_banana`, `fresh_mango`,                `fresh_orange`, `rotten_apple`, `rotten_banana`, `rotten_mango`, `rotten_orange`). Most images are studio shots with a plain white background.
  2. fresh_vs_rotten_my_dataset : This is the mini version of our main dataset with folders and classes being in the same format, the photos in it are taken in     different backgrounds and under different lighting to make the model learn differentiating fruits on the basis of different backgrounds.
  3. fresh_vs_rotten_my_test_dataset : Contains the final **test images** through which we do our final testings.

B. 1 python notebook :  
  Contains complete, well-organized, and well-commented code.

C. Old python notebook : 
  Not to be used anywhere, it's just for the reference to the challenges that we faced during our project.


## How to run the app

### Option 1 : Using the app

| Tab | What it does |
|---|---|
| **Upload** | Upload a JPG or PNG photo of a fruit |
| **Camera** | Take a photo with your device camera |

The result shows **Fruit** and **Status**, followed by a confidence bar for each.

For best results: put **one fruit** in the centre of the photo, fairly close, with good lighting and a plain background.

### Option 2: Using the google drive link 
#### Steps 
1. Go to the drive link : https://drive.google.com/drive/folders/1_98NFG5QR_Ecy4ukxOxm6dFZuIR5Sg8r?usp=sharing
2. Open the google colab notebook named as **"fresh_vs_rotten_fruit_new.ipynb"**
3. Run the cells, check if the drive is mounted properly and the zip file is extracted without any errors.
4. Run all the subsequent cells, train the model.
5. At the last cell's output, a local host link for the app will be created, open it and test.


## How to retrain the model (Google Colab)

You only need this if you change the data or the model. The trained weights are already in this repository.

1. Open `notebooks/fruit_training.ipynb` in Google Colab.
2. Set the runtime to GPU: **Runtime > Change runtime type > T4 GPU**.
3. Upload the dataset zip to your Google Drive and update the paths in the first cells to match your Drive.
4. (Optional) add your own photos to your Drive in the same `train / val / test` layout with the same class folder names, and update the path in the merge cell.
5. Run the cells in order:
   1. Mount Drive and unzip the dataset
   2. Imports and finding the dataset folders
   3. Leakage check and clean split
   4. Merge your own photos
   5. Augmentation and data loaders
   6. Model
   7. Training (about 40 minutes on a Colab T4)
   8. Learning curves
   9. Evaluation (accuracy, F1, per-fruit results, confusion matrices, latency and FPS)
   10. Evaluation on your held-out own photos
   11. Save the model (`fruit_model.pth` and `labels.json`)
6. Copy the two saved files into the repository folder next to `app.py`.
