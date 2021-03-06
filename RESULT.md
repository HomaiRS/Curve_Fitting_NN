I randomly sampled 300 points (xi) from the uniform distribution with mean 0 and variance 1. The goal of this project is to fit di = sin(20xi) + 3xi + νi, (i = 1,..., n) curve using the neural network. We used the backpropagation algorithm with online learning to find the optimal weights/network that minimize the mean-squared error (MSE) (1/n sum(di −f(xi,w))^2). We used different learning rate parameter, and plotted the number of epochs vs the MSE in the backpropagation algorithm.

I also added noise by picking n real numbers uniformly at random on [− 1/10 , 1/10] called vi. I used ϕ(v)=v for the output layer activation function and I used ϕ(v)=tangh(v) for the hidden layers. In this question, I implemented the backpropagation algorithm for online learning using 2 hidden layers that have 24 hidden neurons in each layer and 24 biases (so we have 73 variables that we want to solve for). 

There are several hyperparameters including the learning rate, variance of the distribution used to initialize the weights that needs to be tuned to be able to get a better fit. In my first attempt indicated in Figure 1, the algorithm poorly can approximate the input data, even though the MSE is small and it is about 0.4. The learning rate for this approximation is 0.001. In the following fgure, we showed the fitted curve that looks like a straight line, and poorly approximates the input data.

![Picture1](https://user-images.githubusercontent.com/43753085/104084007-e7605c00-5208-11eb-9d75-a4aaf4ea04f7.png)

Mean square error is computed at each data point and the desired output. It is a monotonically decreasing curve per iteration of the online learning using backpropagation algorithm as idicated in the following figure.

![Picture2](https://user-images.githubusercontent.com/43753085/104083774-ffcf7700-5206-11eb-9c90-d20cea44d24a.png)

However, I changed hyperparameters such as the variance of the distribution that I randomly selected my initialized weights from 1 to 0.2 and I changed the learning rate from 0.001 to 0.01, and the curve has an accurate fit as follows. The following figure shows the fit using learning rate of 0.01 and variance 0.2 instead of 1 for distribution that I initialize the weights.

![Picture3](https://user-images.githubusercontent.com/43753085/104083983-b8e28100-5208-11eb-9e55-c033036e45a3.png)
![Picture4](https://user-images.githubusercontent.com/43753085/104083990-c1d35280-5208-11eb-9fbc-cd898c9a8e58.png)

A better fit was resulted when the eta was changed to 0.02 and the variance of distribution that I initialize the weights from was changed from 1 to 0.2.

![Picture5](https://user-images.githubusercontent.com/43753085/104084080-cd734900-5209-11eb-87e4-94b6d8376d51.png)
![Picture6](https://user-images.githubusercontent.com/43753085/104084084-d5cb8400-5209-11eb-90e9-07c4d6bdeae2.png)



