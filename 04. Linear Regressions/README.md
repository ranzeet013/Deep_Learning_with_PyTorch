###  Linear Regression 


Linear Regression :
                           I created a linear regression model using PyTorch, where the training data consists of input tensor `x_train` and target tensor `y_train`. These tensors were converted into PyTorch Variables, and a linear regression model was initialized using `nn.Linear(1, 1, bias=True)`. The model was trained using stochastic gradient descent (SGD) optimization with a mean squared error (MSE) loss function. The training process involved predicting outputs, computing the loss, performing backpropagation, and updating the model's parameters in a loop of 300 steps. The training progress was visualized by plotting the scatter plot of input data and target values, along with the evolving model predictions. After training, the learned weight and bias parameters of the model were displayed. Additionally, I visualized the cost function over a range of weights to understand how changes in weight affect the cost. The final trained model was tested on a new input value, and the output was printed.

Link: 
[Linear Regression Model Notebook](https://github.com/ranzeet013/Deep_Learning_with_PyTorch/tree/main/4%2C%20Linear%20Regressions/Linear%20Regression%20Model)



###  Multivarient Linear Regression


Multivarient Linear Regression :
                    I created a multivariate linear regression model using PyTorch, where the dataset consists of records of individual test performances. The input features are stored in the `x_data` variable, and the target variable is stored in the `y_data` variable. These variables are converted into PyTorch Variables. The multivariate linear regression model is initialized using `nn.Linear(3, 1, bias=True)`, where the input dimension is 3 (number of features) and the output dimension is 1. The model is trained using stochastic gradient descent (SGD) optimization with a mean squared error (MSE) loss function. The training process involves predicting outputs, computing the loss, performing backpropagation, and updating the model's parameters in a loop of 2000 steps. The training progress is monitored by printing the cost and predictions every 50 steps. The model's accuracy is evaluated by comparing the predicted values with the actual target values, and the sum and average accuracy are printed.

Link:
[Multivarient Linear Regression Notebook](https://github.com/ranzeet013/Deep_Learning_with_PyTorch/tree/main/4%2C%20Linear%20Regressions/Multivarient%20Linear%20Model)


### Naive Linear Regression


Naive Linear Regression :
             I created a simple linear regression model using PyTorch, where the dataset consists of input features stored in the `x_train` tensor and corresponding target values stored in the `y_train` tensor. These tensors are converted into PyTorch Variables. The model is trained using gradient descent, iterating for 200 steps to adjust the model's weight and minimize the difference between predicted and target values. The learning rate is set to 0.01, and the training progress is visualized through a scatter plot. In each step, the loss is calculated, gradients are computed, and weights are updated. The interactive plot shows how the linear regression line evolves during training, along with the cost, weight, and gradient information at each step.

Link:
[Naive Linear Regression](https://github.com/ranzeet013/Deep_Learning_with_PyTorch/tree/main/4%2C%20Linear%20Regressions/Naive%20Linear%20Regression)
