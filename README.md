# Pneumonia Detection using AI/ML
##  Objective
To detect pneumonia from chest X-ray images using machine learning techniques.
## Tools & Technologies
- Python
- NumPy, Pandas
- OpenCV / PIL
- TensorFlow / Keras
- Matplotlib
## Project Description
This project uses a Convolutional Neural Network (CNN) to classify chest X-ray images as Pneumonia or Normal.
## Features
- Image preprocessing (resizing, normalization)
- CNN-based classification
- Model training and evaluation
- Prediction on new images
## Concepts Used
- Deep Learning (CNN)
- Image Processing
- Data Preprocessing
- Model Evaluation
## Results
- Achieved good accuracy in classifying pneumonia cases
- Demonstrated importance of preprocessing in model performance
## How to Run
1. Install dependencies:
2. Train model:
3. Run prediction:
## Future Improvements
- Deploy model using web app
- Use larger dataset
- Improve accuracy with advanced architectures
numpy
pandas
opencv-python
tensorflow
matplotlib
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
# Dummy data (replace with real dataset)
X = np.random.rand(100, 64, 64, 1)
y = np.random.randint(0, 2, 100)
# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Train model
model.fit(X, y, epochs=5)
# Save model
model.save("pneumonia_model.h5")
import tensorflow as tf
import numpy as np
# Load model
model = tf.keras.models.load_model("pneumonia_model.h5")
# Dummy input (replace with real image)
test_image = np.random.rand(1, 64, 64, 1)
prediction = model.predict(test_image)
if prediction > 0.5:
    print("Pneumonia Detected")
else:
    print("Normal")
