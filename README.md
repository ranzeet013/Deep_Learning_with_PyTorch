
#  Deep Learning Journey with PyTorch

I'm sharing my Deep Learning journey with PyTorch in this repository, featuring clear code and resources concentrated on practical implementations of deep learning techniques using the PyTorch framework.


Day_1 of Deep Learning with PyTorch :

Topics:

Deep Learning with PyTorch : 
           Deep Learning with PyTorch is an interdisciplinary pursuit combining computer science and artificial intelligence, centered on enhancing computers' ability to comprehend and process intricate patterns within large datasets. In my ongoing journey, I am delving into the foundations of Deep Learning using PyTorch. I've engaged in comprehensive learning, translating theoretical concepts into practical implementations. The repository showcases clear code and resources, providing a focused approach to mastering deep learning techniques with the PyTorch framework. Similar to the way I've explored Machine Learning, I aim to simplify the complexities, offering insights and practical implementations that facilitate an learning experience. I'm eager to share the upcoming days of exploration and discovery!

Link: 
[Basics of Pytorch](https://github.com/ranzeet013/Deep_Learning_with_PyTorch/blob/main/1.%20PyTorch%20Basics/Pytorch%20Basic%20.ipynb)



Day_2 of Deep Learning with PyTorch :

Topics:

Variable and Autograds :
         I began by creating tensors and converting them into variables using the `Variable` class from the autograd module. This process enables automatic gradient computation during backpropagation, a fundamental aspect of training neural networks. I accessed the data from variables, toggled gradient tracking on and off, and experimented with the `volatile` parameter for situations where gradient calculation was unnecessary. The code further delved into constructing a computational graph, performing operations on variables `x`, `y`, and `z`, and executing a backward pass to calculate gradients. This hands-on experience clarified the seamless integration of automatic differentiation in PyTorch, enhancing my understanding of its utility in building and training deep learning models.

Link:
[Variable And Autograds in PyTorch](https://github.com/ranzeet013/Deep_Learning_with_PyTorch/blob/main/2.%20Variable%20And%20Autograds/Variable%20And%20Gradient.ipynb)



Day_3 of Deep Learning with PyTorch :

Topics:

Activation Functions in PyTorch :
           I delved into the fundamental concept of activation functions, pivotal components in neural network architecture. I initiated the process by defining a range of input values and converting them into a PyTorch variable. Subsequently, I visualized the outputs of diverse activation functions, each serving a unique purpose in introducing non-linearity to the neural network.

Firstly, I implemented the Rectified Linear Unit (ReLU) activation function, observing its capability to replace negative inputs with zero, effectively addressing the vanishing gradient problem. Following this, I explored the Tanh function, which transforms inputs to values between -1 and 1, introducing symmetry around zero and commonly finding application in recurrent neural networks (RNNs). Then I implemented the Sigmoid function, mapping input values to a range between 0 and 1, making it suitable for representing probabilities. However, I noted its potential vanishing gradient issues for extreme inputs. Lastly, I studied the Softplus function, which produces positive values and resembles ReLU for positive inputs. This function is preferred when the output doesn't need to be bounded. Visualizations generated for each activation function provided a tangible understanding of their characteristics and transformative effects on input values. 

Link:
[Activation Function in PyTorch](https://github.com/ranzeet013/Deep_Learning_with_PyTorch/blob/main/3.%20Activation%20Functions%20In%20PyTorch/Activation%20Functions%20.ipynb)



Day_4 of Deep Learning with PyTorch :

Topics:

Linear Regression :
       I worked on several regression scenarios. Firstly, I implemented a simple linear regression model, creating a dataset and visualizing the training progress. Next, I delved into multivariate linear regression, applying it to a dataset with multiple features. Lastly, I ventured into non-linear regression using a neural network with a ReLU activation function. Throughout these endeavors, I handled model initialization, defined loss functions, and executed training processes. Visualizations were crucial to monitoring the training progress, allowing me to grasp the impact of the models on different regression tasks.

Link:
[Linear Regression in PyTorch](https://github.com/ranzeet013/Deep_Learning_with_PyTorch/tree/main/4%2C%20Linear%20Regressions)



Day_5 of Deep Learning with PyTorch :

Topics:

Non Linear Regression in PyTorch :
            I ventured into non-linear regression using a neural network for a synthetic dataset. The dataset was generated with input values ranging from -1 to 1, and the corresponding output values were obtained by squaring the inputs and adding a slight random noise. The neural network architecture included an input layer, a hidden layer with 20 units activated by the Rectified Linear Unit (ReLU), and an output layer. To train the model, I employed the Adam optimizer with a learning rate of 0.1 and used the Mean Squared Error (MSE) loss function to measure the difference between predictions and target values. Over 200 iterations, the model made predictions, computed losses, performed backpropagation, and updated parameters. Every 10 iterations, I visualized the data points along with the model's predictions, providing insights into the training progress. 

Link:
[Non Linear Regression in PyTorch](https://github.com/ranzeet013/Deep_Learning_with_PyTorch/tree/main/5.%20Non%20Linear%20Regression)



Day_6 of Deep Learning with PyTorch :

Topics:

Optimizers in PyTorch :
           I worked on different optimizers (SGD, Momentum, RMSprop, and Adam) on training a neural network for a quadratic regression task. The dataset consisted of 1000 points generated from a quadratic function with added Gaussian noise. The neural network architecture included a hidden layer with ReLU activation and a final prediction layer. I utilized a TensorDataset and a DataLoader for efficient data management, and the models were trained over multiple epochs. Each optimizer adapted the model parameters to minimize the mean squared error loss. The loss histories for each optimizer were recorded, and the final step involved visualizing the loss values over training steps for a comparative analysis. The plot revealed the convergence behavior of different optimizers, aiding in understanding their performance characteristics during the training process.
           
Link:
[Optimizers in PyTorch](https://github.com/ranzeet013/Deep_Learning_with_PyTorch/tree/main/6.%20Optimizers%20In%20PyTorch)



Day_7 of Deep Learning with PyTorch :

Topics:

MNIST Classification :
      I implemented a neural network for digit classification using PyTorch on the MNIST dataset. Firstly, I imported necessary libraries, including torch for tensor operations, torch.nn for neural network components, and torchvision for dataset handling. Then, I preprocessed the data using transformations and created training and test data loaders.

For building the neural network, I defined a class with three fully connected layers. The model architecture included an input layer (28*28 neurons), a hidden layer (128 neurons), and an output layer (10 neurons for digit classes). I used the CrossEntropyLoss as the loss function and Stochastic Gradient Descent (SGD) as the optimizer.

During training, I ran a loop through multiple epochs, iterating over batches of data. In each iteration, I performed forward and backward passes, calculated the loss, and updated the model's parameters. After training, I evaluated the model on the test dataset, calculating accuracy by comparing predicted and actual labels.

The accuracy represented the percentage of correctly classified digits, and the model achieved satisfactory results. This script demonstrated the end-to-end process of building, training, and testing a neural network for digit recognition using PyTorch.

Link:
[MNIST Classification](https://github.com/ranzeet013/Deep_Learning_with_PyTorch/tree/main/7.%20MNIST%20Classification)



Day_8 of Deep Learning with PyTorch :

Topics:

Batch Training :
      I created a PyTorch dataset by generating two tensors, `x` and `y`, with linear sequences of values and concatenated them to form a dataset using `Data.TensorDataset`. I then set up a data loader to efficiently handle the data in batches, iterating through three epochs with batch sizes of five and ten for training monitoring. Additionally, I demonstrated loading and batching the MNIST dataset, and modified a pre-trained ResNet-18 model for a new output size. These implementations illustrate the process of creating custom datasets, utilizing data loaders for efficient training, and working with pre-trained models in PyTorch.

Link:
[Batch Training](https://github.com/ranzeet013/Deep_Learning_with_PyTorch/tree/main/8.%20Batch%20Training)



Day_9 of Deep Learning with PyTorch :

Topics:

DenseNet :
          I created a PyTorch module named DenseBlock to implement a dense block within a neural network, particularly useful for DenseNet architectures. The dense block consists of two convolutional layers with batch normalization and ReLU activation. The first convolutional layer increases the channel dimension by a factor of 4, while the second reduces it to the specified rate. In the forward method, I applied these operations sequentially to the input, resulting in the final output. Additionally, I created another module called TransBlock to facilitate transitioning between different stages of feature maps in a convolutional neural network. This block involves batch normalization, a 1x1 convolutional layer for adjusting channel dimensions, and average pooling for downsampling spatial dimensions. The forward method executes these operations on the input, producing the transformed output. These modules enhance the flexibility and functionality of neural network architectures,

Link:
[DenseNet in PyTorch](https://github.com/ranzeet013/Deep_Learning_with_PyTorch/tree/main/9.DenseNet)



Day_10 of Deep Learning with PyTorch :

Topics:

ResNet :
        I created a PyTorch module named BasicBlock to define a basic building block for a residual network (ResNet) in a convolutional neural network (CNN). The block consists of two convolutional layers, each followed by batch normalization and ReLU activation. The first convolutional layer has a kernel size of 3x3 and a specified stride, while the second convolutional layer has a fixed kernel size of 3x3 with a stride of 1. The block includes a residual connection to facilitate the flow of gradients during training, helping with the vanishing gradient problem. The forward method applies these operations sequentially, and the output is obtained by adding the residual connection to the processed input.

Link:
[ResNet](https://github.com/ranzeet013/Deep_Learning_with_PyTorch/tree/main/10.ResNet)



