deep-learning-challenge
 May 2025 | Dotan Barak


In this project, we leverage deep learning to build a binary classification model that predicts whether an organization funded by Alphabet Soup will be successful. Using machine learning techniques and neural networks, we analyze a dataset of over 34,000 previously funded organizations. The dataset includes various metadata features—such as application type, organizational classification, and use case—which serve as predictors for future funding outcomes.


Data Preprocessing
The dataset underwent a series of preprocessing steps to prepare it for modeling:

Column Reduction: Non-essential and identifier columns were removed to reduce dimensionality and eliminate irrelevant information.

Categorical Encoding: Categorical features were encoded using:

Label encoding for binary categories

One-hot encoding for multi-class nominal variables

Low-Frequency Value Handling: Categories with low occurrence were grouped into an "Other" category to reduce noise and prevent overfitting.

Feature Scaling: A StandardScaler was applied to normalize the input features, ensuring that all features contribute equally during model training.

After preprocessing, the dataset was transformed into a format compatible with machine learning models. The final number of features was determined by:

python
X_train_scaled.shape[1]


Model Architecture
The neural network was designed as a binary classification model with the following architecture:

Input Layer: Configured with the number of features derived from X_train_scaled.shape[1]

First Hidden Layer:

80 neurons

ReLU activation function

Second Hidden Layer:

30 neurons

ReLU activation function

Output Layer:

1 neuron

Sigmoid activation function to produce a probability score for binary classification

This architecture balances model complexity with efficiency, allowing it to learn non-linear patterns while minimizing overfitting.


The output layer consists of a single neuron with a sigmoid activation function, which is well-suited for binary classification problems such as predicting the likelihood of success (encoded as 0 or 1). This activation function outputs a probability between 0 and 1, enabling clear classification thresholds.


 Model Performance
The model was trained for 20 epochs, during which it optimized its weights using binary cross-entropy loss and an accuracy metric. After training, evaluation on the test dataset yielded the following results:
python
Loss: 0.5586
Accuracy: 0.729
This corresponds to an overall test accuracy of approximately 72.29%, indicating reasonably good performance for predicting binary outcomes based on the input features.


The model was compiled using binary cross-entropy (loss="binary_crossentropy"), which is the standard loss function for binary classification problems. It measures the dissimilarity between predicted probabilities and actual binary labels, making it ideal for evaluating performance in tasks with two possible outcomes.


Model Export
The final trained model was saved using:

python
nn.save("charity_model.h5")
This command creates an HDF5 (.h5) file that stores the complete model, including its architecture, trained weights, and optimizer configuration. This format allows for efficient reloading of the model for inference or additional training.

Summary of Model Results
The neural network achieved an accuracy of approximately 73% on unseen test data. This indicates strong generalization performance, especially given the structured nature of the dataset and the preprocessing steps involved.

This model serves as a solid baseline for identifying potentially successful funding applications, made possible by:

Effective feature engineering, including one-hot encoding and consolidation of low-frequency categories

Feature scaling to standardize the input space for learning

Further performance improvements may be achievable through:

Hyperparameter tuning (e.g., number of nodes, hidden layers, epochs)

Regularization techniques, such as dropout or batch normalization, to reduce overfitting and improve generalization

Alternative Model Consideration
To further explore modeling options, an alternative such as a Random Forest Classifier may be beneficial. Random forests are ensemble-based models known for their robustness and efficiency, particularly on tabular datasets like this one.

Advantages of using a Random Forest:

Naturally handles non-linear relationships and interactions with minimal tuning

Provides feature importance metrics for better interpretability

Typically resistant to overfitting due to built-in bagging and ensembling

May outperform shallow neural networks on structured, categorical-heavy data

Exploring a random forest approach could provide both faster training and stronger interpretability, offering a valuable comparison to the neural network baseline.

**References**
IRS Tax Exempt Organization Search – Bulk Data Downloads
https://www.irs.gov/charities-non-profits/tax-exempt-organization-search-bulk-data-downloads
Primary data source for nonprofit organization applications.

TensorFlow – An end-to-end open-source platform for machine learning.
https://www.tensorflow.org/

Keras – High-level API for building and training deep learning models in TensorFlow.
https://keras.io/

scikit-learn – Python module for machine learning, including tools for model evaluation, preprocessing, and traditional ML algorithms.
https://scikit-learn.org/

Pandas – Data analysis and manipulation tool for structured data.
https://pandas.pydata.org/

NumPy – Core library for numerical operations in Python.
https://numpy.org/

