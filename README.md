# Neural-network-pruning-example
# Project README

## MNIST Classification with TensorFlow and Pruning

This project demonstrates how to build and train a neural network to classify handwritten digits from the MNIST dataset using TensorFlow and Keras, with an emphasis on model pruning for optimization.

### Getting Started

1. **Environment Setup**: Ensure you have Python and TensorFlow installed. TensorFlow Model Optimization Toolkit is also required for pruning. You can install it using pip:

   ```shell
   pip install tensorflow tensorflow-model-optimization
   ```

2. **Dataset**: The project uses the MNIST dataset, which is automatically downloaded using TensorFlow's Keras API.

3. **Model Architecture**: A simple Sequential model with two Dense layers is used. The first Dense layer has 512 units and 'relu' activation, and the second Dense layer has 10 units with 'softmax' activation for classification.

4. **Training**: The model is trained with 'rmsprop' optimizer and 'sparse_categorical_crossentropy' loss function.

5. **Pruning**: The TensorFlow Model Optimization Toolkit is utilized to apply pruning to the model. Pruning is configured to gradually increase sparsity from 50% to 80% using a polynomial decay schedule.

6. **Evaluation**: The model's accuracy is evaluated on the test set before and after pruning to compare the performance.

### Instructions

- Load the MNIST dataset and preprocess the data.
- Define and compile the model.
- Train the model with the training data.
- Evaluate the baseline model's performance on the test set.
- Apply pruning to the model and retrain it.
- Evaluate the pruned model's performance and compare it with the baseline model.
- Save the pruned model's weights for further use.

### Key Files

- `model_training.py`: Script for model training and evaluation.
- `model_pruning.py`: Script for applying pruning to the model.

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- TensorFlow Model Optimization Toolkit

### Installation

Clone this repository and navigate to the project directory. Install the required packages using the following command:

```shell
pip install -r requirements.txt
```

### Running the Project

To train and evaluate the model, run:

```shell
python model_training.py
```

To apply pruning to the model and evaluate its performance, run:

```shell
python model_pruning.py
```

### Results

The project includes logging and visualization of training and pruning processes. You can view the accuracy and loss metrics, as well as the effect of pruning on model size and performance.
