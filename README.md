# 🐶 Dog Breed Identification

A deep learning project for classifying dog breeds from images using transfer learning and a user-friendly Streamlit web app.  
Developed and maintained by [aravinds-py](https://github.com/aravinds-py).

---

## 🚀 Features

- **Classifies 120+ dog breeds** using a fine-tuned InceptionV3 model
- **Streamlit web app**: Upload a dog photo and get an instant breed prediction
- Advanced data augmentations and data pipeline with TensorFlow
- **Model files (.keras, .h5)** and label encoder included for immediate use
- Comprehensive Jupyter notebook for data exploration and model building

---

## 📁 Directory Structure

```
├── app/                         # Streamlit app (main entry: app.py)
├── train/                       # Training images (by breed)
├── test/                        # Test images
├── labels.csv                   # Image IDs and breed names
├── sample_submission.xlsx       # Example predictions format
├── dog_breed_classifier.keras   # Saved Keras model
├── dog_breed_classifier.h5      # Saved HDF5 model
├── label_encoder.pkl            # Saved sklearn LabelEncoder
├── Dog Breed Classifier.ipynb   # Jupyter notebook for EDA & training
├── dog-breed-identification.zip # Zipped dataset (optional)
├── README.md                    # This file
```

---

## 🗂️ Dataset

**[👉 Download the dataset here (Google Drive)](https://drive.google.com/file/d/14of-v7y9Q95fqBvOXfTrEg2C2LZ-aq6l/view)**

- Extract the dataset so that the `train/`, `test/`, and `labels.csv` files are in your project directory as shown above.

---

## 🏁 Getting Started

1. **Clone the repo**
    ```bash
    git clone https://github.com/aravinds-py/dog-breed-identification.git
    cd dog-breed-identification
    ```
2. **Install requirements**
    ```bash
    pip install tensorflow streamlit opencv-python-headless pandas scikit-learn matplotlib seaborn
    ```
3. **Download and extract the [dataset](https://drive.google.com/file/d/14of-v7y9Q95fqBvOXfTrEg2C2LZ-aq6l/view)**
4. **Run the Streamlit app**
    ```bash
    streamlit run app/app.py
    ```
5. **Upload an image** through the web interface to get the predicted dog breed!

---

## 🧑‍💻 Usage

- Use **`Dog Breed Classifier.ipynb`** for full step-by-step exploration, training, augmentation, and saving models/encoders.
- The **`app/`** folder contains all code for the Streamlit web UI.
- Model inference on new images is supported via both the app and direct Python (`load_model` + `label_encoder.pkl`).

**Sample inference code:**
```python
import tensorflow as tf, pickle, numpy as np, cv2
model = tf.keras.models.load_model("dog_breed_classifier.keras")
with open("label_encoder.pkl", "rb") as f: le = pickle.load(f)
def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)
img = preprocess_image("your_dog_photo.jpg")
pred = model.predict(img)[0]
idx = np.argmax(pred)
print("Predicted breed:", le.inverse_transform([idx])[0])
```

---

## 📊 Data & Training

- Training/validation split and label encoding with `scikit-learn`.
- Model: Transfer-learned [InceptionV3](https://keras.io/api/applications/inceptionv3/), top layers custom, trained on resized images.
- All files needed for inference (model, encoder, labels) are included in the repo structure.

---

## 📦 Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [Streamlit](https://streamlit.io/)
- [Kaggle Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification/data)
- [scikit-learn](https://scikit-learn.org/)

---

## 📝 License

[MIT](LICENSE)

---

*Made with ❤️ by [aravinds-py](https://github.com/aravinds-py)*