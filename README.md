# CIFAR-10 Image Classification with CNN

This repository contains a Python script for building and training a CNN model to classify images from the CIFAR-10 dataset.

## Getting Started

### 1. Clone the repository:
```gitbash
git clone https://github.com/kevintian4/Cifar10-Image-Recognition.git
```
### 2. Install dependencies:
```gitbash
pip install tensorflow keras matplotlib numpy
```

### 3. Running the script:
The project consists of a single script (cifar10.py) that performs the following tasks:

- Loads the CIFAR-10 dataset.
- Preprocesses the data (normalization).
- Defines and builds the CNN model architecture.
- Trains the model on the training data.
- Evaluates the model's performance on the test data.
- Visualizes sample predictions (optional).
<p>To run the script and train the model, execute the following command in your terminal:</p>
  
```gitbash
python cifar10.py
```

## Project Structure
<ul>
  <li><code>cifar10.py</code> Script containing the core functionalities for training, evaluation, and visualization.</li>
  <li><code>cifar10_model.h5</code> This file will be created after running the script and stores the trained model weights.</li>
</ul>
