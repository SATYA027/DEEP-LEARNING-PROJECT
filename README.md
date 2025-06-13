# DEEP-LEARNING-PROJECT

COMPANY: CODTECH IT SOLUTIONS

NAME: M.D.S.D.Satyanarayana Reddy

INTERN ID: CT06DL1428

DOMAIN: DATA SCIENCE

DURATION: 6 WEEKS

MENTOR: NEELA SANTHOSH


##DESCRIPTION:

Task 2: Sentiment Analysis Using TensorFlow and IMDB Dataset

The task is to build a binary classification model that can determine whether a movie review is positive or negative. For this, we use the IMDB dataset, a popular dataset in natural language processing (NLP), which contains 50,000 preprocessed movie reviews. The dataset is split into 25,000 reviews for training and 25,000 for testing, each labeled as either 0 (negative) or 1 (positive).

** Step 1: Data Loading and Preprocessing

To make the model efficient, only the top 10,000 most frequent words in the dataset are used. Each review is represented as a list of integers, where each integer corresponds to a specific word index in the dataset's vocabulary.
Since neural networks require inputs of fixed length, all reviews are standardized by padding or truncating them to 200 words. If a review is shorter than 200 words, zeros are added at the beginning. If it is longer, only the last 200 words are kept. This ensures that every input sequence has the same shape, which is essential for feeding into a deep learning model.

** Step 2: Model Architecture

The model is built using a sequential neural network, meaning each layer flows directly into the next.The first layer is an Embedding layer. This converts each word index into a dense vector of fixed size (in this case, 32 dimensions). Instead of using raw integers, the embedding allows the model to learn more meaningful representations for words, capturing semantic relationships (like "good" being closer to "great" than to "bad").
Next is the Global Average Pooling layer, which takes the average of all the word vectors in a review. Instead of treating each word separately, this layer summarizes the entire review into a single vector. This is a simple yet effective way to reduce dimensionality and make the model faster and easier to train.
Following this is a Dense (fully connected) layer with 16 units and a ReLU activation function. This layer helps the model learn complex patterns and non-linear relationships in the data. It processes the summarized review vector and learns features that are useful for distinguishing between positive and negative sentiments.
The final layer is another Dense layer with a single output unit and a sigmoid activation function. The sigmoid function maps the output to a value between 0 and 1, which represents the predicted probability that the review is positive. A value closer to 1 means the model thinks the review is positive; closer to 0 means negative.

**Step 3: Compilation and Training

Before training, the model is compiled by specifying three main components:
Optimizer: Adam is used, which is adaptive and efficient for most deep learning tasks.
Loss function: Binary crossentropy is chosen because this is a binary classification problem.
Metrics: Accuracy is used to evaluate how many predictions the model gets right.
The training data is further split into a training set and a validation set. The first 10,000 samples are used for validation, while the remaining 15,000 are used to train the model. This allows us to monitor the model’s performance on unseen data during training and helps detect overfitting.
Training is done over 5 epochs, meaning the model goes through the training data 5 times. A batch size of 512 is used, meaning the model updates its weights after processing every 512 samples. This balances memory efficiency and model performance.

**Step 4: Evaluation

After training, the model is evaluated on the test set, which contains completely unseen data. The evaluation returns the loss and accuracy of the model on the test set. Accuracy shows how well the model can generalize to new reviews.


##OUTPUT:
![image](https://github.com/user-attachments/assets/065cb995-ae28-4e7a-8b69-def4c3073b57)

