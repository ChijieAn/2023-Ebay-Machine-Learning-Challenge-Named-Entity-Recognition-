# 2023-Ebay-Machine-Learning-Challenge-Named-Entity-Recognition-
We fine tuned a pretrained Bert German NER model on a dataset of 5000 records on the eBay German website. We also tried strategies like bagging and adding a CRF head to the Bert model. 

# Data processing and training folder
In this folder, there're three python files

In data_processing_functions.py, I defined some functions to help process the given training data, implementing the BOI labeling strategy, as well as defining a dataset class for later training.

In training_functions.py, I defined some functions to help with the training and evaluations process. It also includes a CRF head class, which can be added to the original Bert model to adjust the structure of the model. 

In data_processing_training.py, I imported the functions in the other two python files to process the data and train the Bert model.
