# Autoencoding MNIST digits

Explore dense and sparse encodings of digit images.

## Autoencoders

Autoencoding is an unsupervised learning problem, training a neural network to reproduce its input as accurately as possible.  A network trained for this can be used to "encode" an input sample by storing the activations at one layer of the network, and decode that encoding by passing it through all the subsequent layers.  

Given networks where each layer has more representational capacity than the input input, without additional constraints, minimizing squared reconstruction error will result in a network that learns to copy the input to the output directly, achieving perfect accuracy but not *encoding* the image in any interesting way.  The encoding function is essentially the identity function.

Desigining an autoencoder involves considering the properties you want your code vectors to have and how to constrain your network and/or training procedure so an encoding with those properties is found.

### Finding dense codes for MNIST

One constraint to prevent that copymachine is to create an "information bottleneck" in the network architecture, where at least one of the hidden layers has fewer units than the number of input dimensions.  A network this narrow cannot learn to copy its input because it isn't large enough to store a whole input sample. An autoencoder with fewer units in the code layer than input dimensions (the bottleneck) learn to create "dense" codes, vectors of floating point numbers to whose precise values the output is highly sensitive.  

This project creats a multilayered autoencoder to encode a 28x28 pixel grayscale image into a vector and to decode those vectors back into images.  The usual squared error loss function is minimized to find the weights:

For sample vector $x$, let $E(x)=c$ be the encoding function computed by the "first half" of the network, the activation vector of the code layer when the network sees input vector $x$.  Let let $D(c) = x'$ be the decoding function, the "second half" of the network that transforms code vector $c$ back into the reconstructed image $x'$.  The loss function for training set $X$ is:

$$
L(X) = \sum_{x\in X} (x - D(E(x)))^2
$$

Run the script `> python dense.py` to train an autoencoder on MNIST digits.  Editing the file near the bottom defines the network and training process:

```
def dense_demo():
    de = DenseExperiment(enc_layers=(256, 64,), n_epochs=50)
```

The script trains an autoencoder with 256 hidden units and 64 code units. The parameter `enc_layers` defines the encoders layer sizes and the decoder has the reverse structure.  It also plots the distribution of MSE error in the test set (1,000 samples per digit) and 39 example encoded & decoded images from 8 different regions in this distribution (centered around the 8 quantiles):

![dense result](/assets/dense_64.png)

This show all but the highest error samples reproduce well after being compressed from 784 pixel values to 64 activations and then decoded back to images and that the images with the lowest reproduction error are all of the digit 1.

Also plotted are the "difference images", showing for each of the test digits which pixels in the reconstruction shouldn't be active (in red), and which pixels should be but aren't (shown in blue):

![dense result](/assets/dense_diffs.png)


### Finding sparse binary codes

What if we want to encode the images as sparse (mostly zero) binary vectors instead of all nonzero floating-point values?  We need the following modifications:
* The middle (code) layer should have an activation function that is at least roughly binary.  (If not actually binary, at inference-time this can be thresholded to create the binary encoding.)
* There can be more code bits than pixels since most of them will be zero for a given image.  This sparsity provides the constraint preventing the network from copying the input into the output with no generalizing/re-representation since the whole image can't be stored in a few bits.  
* The loss function should reward codes with fewer 1's (more sparse).

#### Binary
To make the codes binary, the code layer units use the binary / Heaviside activation function with constant pseudoderivative for training gradients:
* $f(x) = 0 \text{ if } x<0 \text{, else } 1$
* $f'(x) = 1$

(In implementation, this is a layer with normal sigmoid units that each feed 1-to-1 into a layer of binary thresholding units with pseudoderivatives.)

#### Sparse
To encourage code sparsity, the loss function has a regularization term, the total number of active bits in the encoded input. 

$$
L(X) = \sum_{x\in X} (x - D(E(x)))^2  + \sum E(x)
$$

Also called the L1 norm.  (A parameter is implemented to control the relative importance of the two terms, but doesn't seem to matter in practice.)

Run the script `> python sparse.py` to train a sparse binary autoencoder with an encoder with 512, 128, and 4096 units in its three layers, producing encodings of 4096 bits.  This plots the same figures as the dense autoencoder example and two additional plots:

First, a plot of the error rates, sparsity, and sample encoded/decoded digits:

![sparsity](/assets/sparse_binary_L1-reg.png)

This shows the distributions of reconstruction error for the different digits in the upper left, the number of active bits per encoded image for the 10 classes of images in the lower left, and a comparison between original and reconstructed images on the right. 

Also plotted is an image showing the encoding of 100 sample digits from each of the 10 digit classes. The code bits are sorted by how often they are active in the test sample in descending order from left to right, and samples from the 10 digits are grouped vertically:

![sparse codes](/assets/sparse_codes_full.png)

Interestingly, some of the code bits are always on (columns of all white pixels on the far left).  Since they cannot be useful in distingusing between digits, they might be eliminated with further training to reduce the regularization term of the loss function.  Also notable is that the vast majority of code bits are never used, for *any* digit.  Any contribution they could provide to reducing MSE error is offset by increasing the regularization term value.

Zooming in to the interesting region shows encodings of the same digits are similar to each other, different from encodings of different digit images:

![sparse codes](/assets/sparse_codes.png)


### Misc observations about sparsity, loss fucnction and binary code units:

With the `L1` and `L1-squared` loss functions on binary units (tanh + binary/Heaviside pass-through layers at the end of the encoding stage), the training process pressures the network to find code words with fewer active bits.

With the `entropy` and `entropy-squared` loss functions (using real-valued/sigmoid units for the code layer),  


##### File notes

* `pca.py` - unit tests / plot tests
* `pca_digits.py` - reconstruction loss of mnist (generates figures)
* `test_embeddings.py` plot tests
* `embed.py` - Generate an embedding from a model's latent representation of MNIST, make a PNG of it, save it.  ALso generate figures for raw PCA(2)
* `draw_tiles.py` - Speed tests

##### References

* ["Cyclical Annealing Schedule:  A Simple Approach to Mitigating KL Vanishing"](https://arxiv.org/pdf/1903.10145),  Hao Fu, Chunyuan Li, Xiaodong Liu, Jianfeng Gao, Asli Celikyilmaz, Lawrence Carin, 2019.
* ["Typography-MNIST (TMNIST): an MNIST-Style Image Dataset to Categorize Glyphs and Font-Styles"](https://arxiv.org/abs/2202.08112), Nimish Magre and Nicholas Brown, 2022.