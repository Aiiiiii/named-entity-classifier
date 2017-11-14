# named-entity-classifier


In this repo, we train a word2vec model, then use it to improve the named-entity classifier from a Logistic Regression.


We also use TensorFlow to implement a Recurrent Neural Network for the Named Entity Recognition problem.  
- Use dimension 20 for the hidden layer of each state
- Use cross-entropy as the loss function
- Use the logistic function as the activation function for the hidden state, and use softmax to product the output for each time step.
