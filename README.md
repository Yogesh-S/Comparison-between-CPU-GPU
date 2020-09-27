# Comparison-between-CPU & GPU

While using a **really deep** neural network. If you try to train this on a CPU like normal, it will take a long, long time. Instead, we're going to use the GPU to do the calculations. The linear algebra computations are done in parallel on the GPU leading to 100x increased training speeds. It's also possible to train on multiple GPUs, further decreasing training time.

PyTorch, along with pretty much every other deep learning framework, uses [CUDA](https://developer.nvidia.com/cuda-zone) to efficiently compute the forward and backwards passes on the GPU. In PyTorch, we move our model parameters and other tensors to the GPU memory using `model.to('cuda')`. We can move them back from the GPU with `model.to('cpu')` which we'll commonly do when we need to operate on the network output outside of PyTorch. As a demonstration of the increased speed, we can see the commparison on how long it takes to perform a forward and backward pass with and without a GPU.
