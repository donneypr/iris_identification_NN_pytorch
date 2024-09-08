Iris Identification using Neural Networks in PyTorch
This repository contains a simple neural network model built using PyTorch to classify iris species based on the well-known Iris dataset. The dataset consists of 150 samples from each of three species of Iris flowers (Iris setosa, Iris virginica, and Iris versicolor), with 50 samples for each species. Each sample has four features: sepal length, sepal width, petal length, and petal width.

Project Overview
The goal of this project is to implement a neural network model that can accurately classify iris species based on the four features using PyTorch. The model architecture consists of two hidden layers and uses the ReLU activation function for non-linearity. The model is trained using the Adam optimizer and the Cross-Entropy loss function.

After training, the model is evaluated on unseen test data, and its performance is measured based on accuracy. The trained model can also be used to make predictions on new data.

Model Architecture
Input Layer: 4 features (sepal length, sepal width, petal length, petal width)
Hidden Layer 1: 8 neurons
Hidden Layer 2: 9 neurons
Output Layer: 3 output neurons (one for each iris species)
Activation function: ReLU

Loss function: CrossEntropyLoss

Optimizer: Adam (learning rate = 0.01)

Dataset
The Iris dataset used in this project is available publicly from the UCI Machine Learning Repository. It consists of 150 samples with 4 features (sepal length, sepal width, petal length, and petal width) and a corresponding label indicating the species.

Features
Sepal Length (cm)
Sepal Width (cm)
Petal Length (cm)
Petal Width (cm)
Labels
0: Setosa
1: Versicolor
2: Virginica
Training the Model
The dataset is first split into training and testing sets (80% training, 20% testing).
The neural network is trained over 100 epochs, and the loss is recorded at each step to monitor the learning process.
After training, the model is evaluated on the test set to measure its accuracy.
Performance
The model achieves 100% accuracy on the test dataset during evaluation.
Predictions for new flower data can be made using the trained model.

![image](https://github.com/user-attachments/assets/8e86592d-882d-445f-b70a-a3165ddb7502)

Usage
1. Clone the Repository

git clone https://github.com/donneypr/iris_identification_NN_pytorch.git
cd iris_identification_NN_pytorch

2. Install Dependencies
Install the necessary Python packages:

pip install torch pandas matplotlib scikit-learn

3. Run the Notebook
The project is implemented in a Jupyter notebook. You can run the notebook interactively to train the model and test it:

jupyter notebook simple_neural_network_identify_iris.ipynb

4. Load and Test the Pre-trained Model
You can save and load the model's state_dict for future use:

# Save the model
torch.save(model.state_dict(), 'my_iris_model.pt')

# Load the model
new_model = Model()
new_model.load_state_dict(torch.load('my_iris_model.pt'))
new_model.eval()

5. Predict on New Data
You can also make predictions for new iris flower data:

new_flower = torch.tensor([5.6, 3.7, 2.2, 2.2])
with torch.no_grad():
    print(new_model.forward(new_flower))

Conclusion
This project demonstrates a simple yet effective implementation of a neural network for classification using PyTorch. With proper tuning and training, the model can achieve excellent performance on the Iris dataset and can be further extended to handle other classification tasks.
