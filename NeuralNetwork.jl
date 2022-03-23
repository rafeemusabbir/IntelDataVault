### A Pluto.jl notebook ###
# v0.18.0

using Markdown
using InteractiveUtils

# ╔═╡ b8b8f9d2-0f09-4c82-aeb9-33b1ca0cc71e
begin
    import Pkg 
    Pkg.activate(mktempdir()) 
    Pkg.add([   
		Pkg.PackageSpec(name="PlutoUI"),
		Pkg.PackageSpec(name="Plots"),
		Pkg.PackageSpec(name="Random"),  
		Pkg.PackageSpec(name="LinearAlgebra"),  
		Pkg.PackageSpec(name="Distributions"),
        Pkg.PackageSpec(name="MLDataUtils"),
        Pkg.PackageSpec(name="Symbolics"),
		Pkg.PackageSpec(name="DataFrames"),
		Pkg.PackageSpec(name="Statistics"),
        Pkg.PackageSpec(name="Measures"),
		Pkg.PackageSpec(name="CSV"),
		Pkg.PackageSpec(name="MLDatasets"),
		Pkg.PackageSpec(name="RDatasets"),
        Pkg.PackageSpec(name="LaTeXStrings"),
		Pkg.PackageSpec(name="Flux"),
		Pkg.PackageSpec(name="Metalhead"),
		Pkg.PackageSpec(name="MLJBase"),
		Pkg.PackageSpec(name="StableRNGs")
    ])
	
	using Flux, Metalhead
    using Symbolics, PlutoUI, LaTeXStrings
	using DataFrames, Random, LinearAlgebra
	using CSV
	using Plots
	using Markdown
	using StableRNGs
end

# ╔═╡ 5fda3c7f-2045-41ea-af20-ef2a987fa0c7
begin
	using InteractiveUtils
	with_terminal() do
		versioninfo()
	end
end

# ╔═╡ bd97eae6-e049-4404-929d-5505e46b3b69
html"""
	<div>Happiness, Peace, and Divine ! Take care of yourself and your loved ones!</div>
<div id="snow"></div>
<style>
	body:not(.disable_ui):not(.more-specificity) {
        background-color:#DCE5E1;
    }
	pluto-output{
		border-radius: 0px 8px 0px 0px;
        background-color:#DCE5E1;
	} 
	#snow {
        position: fixed;
    	top: 0;
    	left: 0;
    	right: 0;
    	bottom: 0;
    	pointer-events: none;
    	z-index: 1000;
	}
</style>
<script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
<script>
        setTimeout(() => window.particlesJS("snow", {
            "particles": {
                "number": {
                    "value": 50,
                    "density": {
                        "enable": true,
                        "value_area": 800
                    }
                },
                "color": {
                    "value": "#ffffff"
                },
                "opacity": {
                    "value": 0.8,
                    "random": false,
                    "anim": {
                        "enable": false
                    }
                },
                "size": {
                    "value": 5,
                    "random": true,
                    "anim": {
                        "enable": false
                    }
                },
                "line_linked": {
                    "enable": false
                },
                "move": {
                    "enable": true,
                    "speed": 3,
                    "direction": "bottom",
                    "random": true,
                    "straight": false,
                    "out_mode": "out",
                    "bounce": false,
                    "attract": {
                        "enable": true,
                        "rotateX": 300,
                        "rotateY": 1200
                    }
                }
            },
            "interactivity": {
                "events": {
                    "onhover": {
                        "enable": false
                    },
                    "onclick": {
                        "enable": false
                    },
                    "resize": false
                }
            },
            "retina_detect": true
        }), 3000);
	</script>
"""


# ╔═╡ 023b3c90-03a0-11ec-3ec9-b5090a456e13
md"""
[A little bit about myself:](#A little bit about myself)

* **Sarder Rafee Musabbir**, PhD Candidate 🌐 University of California davis
* Autonomous Vehicle, Dynamic Games, and Data Science Enthusiast trained in Traffic Operation & Control, Optimal Control, Statistics, Machine Learning 🚀
* You can find me on [Github](https://github.com/rafeemusabbir) or on [LinkedIn](https://www.linkedin.com/in/sarder-rafee-musabbir-95446a73/)
"""

# ╔═╡ f01ef8f2-1032-443c-9782-87d815ce3d2b
md"""
Notebook Source: [Medium Blog](https://towardsdatascience.com/how-to-build-an-artificial-neural-network-from-scratch-in-julia-c839219b3ef8)
"""

# ╔═╡ bb3cd33f-7bb4-47a2-ba88-5f4c60e69b39
md"""
# What is a Neural Network?
"""

# ╔═╡ abab9e4f-0d1e-40f7-ab19-87701d767b78
md"""
At some basic level, Neural networks can be seen as a system which tries to take inspiration from how biological neurons share information with each other in an attempt to discover patterns within some supplied data. Neural networks are typically composed of interconnected layers. The layers of a vanilla Neural Network are:
+ Input Layer
+ Hidden Layer(s)
+ Output Layer(s)
"""

# ╔═╡ e8aa1fc8-3394-4c66-a641-607d1e38a88c
md"""
The input layer accepts the raw data which are passed onto the hidden layer(s) whose job is to try to find some underlying patterns in this raw data. Finally, the output layer(s) combines these learned patterns and spits out a predicted value.
"""

# ╔═╡ 61c9d2d8-42c9-4d35-b9d6-73e8b5fab586
md"""
**Gross Overview of a Neural Network**$(LocalResource("D:\\Materials_and_Documents\\Research_and_Publication\\Programming\\Julia\\tutorial\\Neural_Network\\images\\neural_network_intro1.png"))
"""

# ╔═╡ e83cedb6-bc03-467b-bb4f-97f64e353769
md"""
How do these interconnected layers share information? At the heart of neural networks is the concept of neurons. All these layers have one thing in common, they have are composed of neurons. These neurons are responsible for the transmission of information across the network but what sort of mechanism is used for this data transmission?
"""

# ╔═╡ 6fa60d30-dbe7-4bc1-bf0f-fb5a8a809daa
md"""
# Neurons — The Building Block Of Neural Networks
"""

# ╔═╡ 3106fec3-e5a5-4ac1-af88-fe91e23574b5
md"""
Instead of pouring out some machine learning theories to explain this mechanism, let us simply explain it with a very simple yet practical example involving a single neuron. Assuming that we have a simple case where we want to train a neural network to distinguish between a dolphin and a shark. Luckily for us, we have thousands or millions of information gathered from dolphins and sharks around the world. For simplicity’s sake, let’s say the only variables recorded for each of the observed are `size(X1)` and `weight(X2)`. Every neuron has its own parameters — `weights(W1)` and `bias(b1)` — which are used to process the incoming data `(X1 & X2)` with these parameters and then share the output of this process with the next neuron. This process is visually represented below;
"""

# ╔═╡ d090a450-5cdf-4b89-9e2e-5709613f4724
md"""
**Neuron Activation**
$(LocalResource("D:\\Materials_and_Documents\\Research_and_Publication\\Programming\\Julia\\tutorial\\Neural_Network\\images\\neuron1.jpeg"))
"""

# ╔═╡ 64aa304b-fc5e-4033-967a-7aea84c9082d
md"""
A typical network is composed of so many layers of neurons (billions, in the case of GPT-3). The aggregate of these neurons make up the main parameters of a vanilla neural network. Ok, we know how a neuron works, but how does an army of neurons learn how to make good predictions on some supplied data?
"""

# ╔═╡ a9d7f452-3df6-4834-9764-039d85531c50
md"""
# Vanilla NN: Main Components
"""

# ╔═╡ 8a60aeb1-45a1-4203-ae2c-782bd1032416
md"""
To understand this learning process, we will compose our own neural network architecture to solve the task at hand. To do this, we would need to breakdown the sequence of operations within a typical vanilla network via 5 components:
"""

# ╔═╡ e01d8ed6-5124-4be9-946e-29bb587ebb72
md"""
**Vanilla Neural Network**
$(LocalResource("D:\\Materials_and_Documents\\Research_and_Publication\\Programming\\Julia\\tutorial\\Neural_Network\\images\\vanilla_Neural1.jpeg"))
"""

# ╔═╡ 0ad489ba-b9b8-4793-86bc-17267cd387f9
md"""
The nature of the problem for which a network is designed to solve normally dictates the range of choices available in these 5 components. In a way, these networks are like LEGO systems where users plug in different activation functions, loss/cost functions, optimization techniques. It is of no surprise that this deep learning is one of the most experimental fields in scientific computing. With all these in mind, let us now move over the fun part and implement our network to solve the binary classification task.
"""

# ╔═╡ ddae3887-a59e-4dd3-94ca-47f90f46623b
md"""
**NB:** The implementation will be vectorized with input data of dimensions (n, m) examples mapping some ouptut target of dimensions (m, 1) where n is the number is the number of features and m is the number of training examples.
"""

# ╔═╡ db370532-3b03-40d7-a07a-732d70545372
md"""
**Vanilla Neural Network**
$(LocalResource("D:\\Materials_and_Documents\\Research_and_Publication\\Programming\\Julia\\tutorial\\Neural_Network\\images\\neural_network_2.png"))
"""

# ╔═╡ d07c30a6-8acf-4dd5-9f94-38730a8be4b2
md"""
## Initialisation of Model Parameters
"""

# ╔═╡ b4c13e29-7e77-4a25-936d-542641e89e52
md"""
These networks are typically composed by stacking up many layers of neurons. The number of layers and neurons are fixed and the elements of these neurons (weights and bias) initially normally populated some random values (for symmetry-breaking since we have an optimisation problem). This snippet randomly initialiases the parameters of a given network dimensions:
"""

# ╔═╡ 0ca007a0-c44b-4858-af07-7abec19bfaed
begin
	
"""
    Function to initialise the parameters or weights of the desired network.
"""
function initialise_model_weights(layer_dims, seed)
    params = Dict()

    # Build a dictionary of initialised weights and bias units
    for l = 2:length(layer_dims)
        params[string("W_", (l-1))] = rand(StableRNG(seed), layer_dims[l], 			 										layer_dims[l-1]) * sqrt(2/layer_dims[l-1])
			
        params[string("b_", (l-1))] = zeros(layer_dims[l], 1)
    end
    return params
end

end

# ╔═╡ c4dffeb7-2dc4-43ed-9cca-040e791fbd37
md"""
## Forward Propagation Algorithm
"""

# ╔═╡ df589864-1d1a-4565-a3a7-d0537d8d9d5e
md"""
With the parameters of the model/network initialised, let us build the functions that will run the input data through the network to generate a predicted value. As explained earlier, activation function functions play a crucial role in this flow of information. Our network will use the ReLU activator for activations upto the output layer and the sigmoid activator. These activators are visualised below;
"""

# ╔═╡ ef9b1adb-ba9a-49f7-b840-3f33cbfeefab
md"""
**Activation Functions**
$(LocalResource("D:\\Materials_and_Documents\\Research_and_Publication\\Programming\\Julia\\tutorial\\Neural_Network\\images\\activation_functions.jpeg"))
"""

# ╔═╡ 547ef676-7baf-4685-bbce-3bb649310274
md"""
These two functions implements these activators;
"""

# ╔═╡ 3c07574a-624a-4d58-918d-99f5d8f73dc0
begin
"""
    Sigmoid activation function
"""
function sigmoid(Z)
    A = 1 ./ (1 .+ exp.(.-Z))
    return (A = A, Z = Z)
end

"""
    ReLU activation function
"""
function relu(Z)
    A = max.(0, Z)
    return (A = A, Z = Z)
end
end

# ╔═╡ 00c74375-bcb6-4ec8-a489-3f62e9b0b682
md"""
With our activators implemented, our attention now switches to the movement of data from the input layer to the output layer via the forward pass or propagation sequence. First, a linear output is calcluated and then one of the activation functions is used to convert the linear output into a non-linear one.
"""

# ╔═╡ a7d21076-9c5b-4147-ad4a-1bec8a5e7954
md"""
**Forward Propogation Vectorized**
$(LocalResource("D:\\Materials_and_Documents\\Research_and_Publication\\Programming\\Julia\\tutorial\\Neural_Network\\images\\forward_propogation1.png"))
"""

# ╔═╡ fefa0375-76ef-4dd4-81d1-9ddb31c67274
md"""
**NB:** The inputs and outputs of the activations are ‘cached’ or saved for later retrieval and usage.
"""

# ╔═╡ 05a2d223-737d-4646-9a45-1e062144f923
md"""
These two functions below provide the utilities for moving the data through the network are;
"""

# ╔═╡ c21216e9-c8dc-48a7-9183-0651a48c79e2
begin
"""
    Make a linear forward calculation
"""
function linear_forward(A, W, b)
    # Make a linear forward and return inputs as cache
    Z = (W * A) .+ b
    cache = (A, W, b)

    @assert size(Z) == (size(W, 1), size(A, 2))

    return (Z = Z, cache = cache)
end


"""
    Make a forward activation from a linear forward.
"""
function linear_forward_activation(A_prev, W, b, activation_function="relu")
    @assert activation_function ∈ ("sigmoid", "relu")
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation_function == "sigmoid"
        A, activation_cache = sigmoid(Z)
    end

    if activation_function == "relu"
        A, activation_cache = relu(Z)
    end

    cache = (linear_step_cache=linear_cache, activation_step_cache=activation_cache)

    @assert size(A) == (size(W, 1), size(A_prev, 2))

    return A, cache
end
end

# ╔═╡ 443476ef-5878-45fa-bc65-60e2066b18a0
md"""
The sequence for whole forward propagation algorithm is pretty straightforward. We simply loop over the network dimensions in such a way that the output of a layer becomes the input of the next layer until the last layer which spits out the a predicted value or score (a probability score when sigmoid activation is used). This snippet below implements this sequence
"""

# ╔═╡ 3cdd0065-5545-4d87-b7ba-973ac29bfed2
begin
"""
    Forward the design matrix through the network layers using the parameters.
"""
function forward_propagate_model_weights(DMatrix, parameters)
    master_cache = []
    A = DMatrix
    L = Int(length(parameters) / 2)

    # Forward propagate until the last (output) layer
    for l = 1 : (L-1)
        A_prev = A
        A, cache = linear_forward_activation(A_prev,
                                             parameters[string("W_", (l))],
                                             parameters[string("b_", (l))],
                                             "relu")
        push!(master_cache , cache)
    end

    # Make predictions in the output layer
    Ŷ, cache = linear_forward_activation(A,
                                         parameters[string("W_", (L))],
                                         parameters[string("b_", (L))],
                                         "sigmoid")
    push!(master_cache, cache)

    return Ŷ, master_cache
end
end

# ╔═╡ 1564c00b-48cc-484c-bc3c-afb47f33404f
md"""
## Cost/Loss Function
"""

# ╔═╡ 989c7a4b-5332-4192-96d1-b00d44f54242
md"""
We now have a network which is capable of making some predictions for our training examples based on the randomly initialised parameters. The next step is to see how good those predictions are. For this objective, the network needs a cost or loss function. This cost/loss function has a simple (crucial) task which is to give a summary of how close the predictions are to the actual output values for all the training examples. Log Loss (Binary Cross Entropy) function is suited for a binary classification problems so this next function implements that cost function;
"""

# ╔═╡ 8791f35a-d73b-43dd-aa95-2a35d41308ad
md"""
**Cross Entropy**
$(LocalResource("D:\\Materials_and_Documents\\Research_and_Publication\\Programming\\Julia\\tutorial\\Neural_Network\\images\\log_loss1.png"))
"""

# ╔═╡ e39a4243-0042-44db-a359-b51e4f46d162
begin
"""
    Computes the log loss (binary cross-entropy) of the current predictions.
"""
function calculate_cost(Ŷ, Y)
    m = size(Y, 2)
    epsilon = eps(1.0)

    # Deal with log(0) scenarios
    Ŷ_new = [max(i, epsilon) for i in Ŷ]
    Ŷ_new = [min(i, 1-epsilon) for i in Ŷ_new]

    cost = -sum(Y .* log.(Ŷ_new) + (1 .- Y) .* log.(1 .- Ŷ_new)) / m
    return cost
end
end

# ╔═╡ ad76b1bc-f4b0-4400-a175-e5d0d8cf2fdf
md"""
## Backpropagation Algorithm
"""

# ╔═╡ e3896c90-cebc-44c0-b6b5-6a0114cf1744
md"""
Our network can generate some random values as predictions and it can also know how far off these predictions are from the acutal outputs so the next logical step will be figuring out how to improve its decision making capabilities beyond making random predictions, right?

At the heart of the ‘learning’ process with neural networks is the backpropagation algorithm. Unsurprisingly, that is the most difficult part for newbies to wrap their heads around. Let’s try to unpack this step with some high level intuition before implementing it in our network.

The goal of the backpropagation sequence is to understand the how each of the parameters (all the weights & biases of all the layers) of the network changes with respect to the cost/loss function. But why would we want to do that? This process allows us to turn the learning strategy into a minimisation objective where we want that cost/loss output to be a low as possible. Additionally, we also know and can measure how changes in the bolts and knobs (weights and biases) of our network affect the loss function. Ponder over this simplified analogy for a second and realise how powerful this concept is! Ultimately, all we have to do is tweak the parameters — weights and biases — of the network such that it delivers the lowest value in terms of the loss/cost function and voila, we are close to make predictions just like in Sci-Fi movies!

At the heart of implementation side of backpropagation, is the concept of chain rule borrowed from multivariate calculus. In fact, the whole sequence for backpropagation can be seen as a long chain of partial derivatives of each of the layers (and their weights ‘W’ and biases ‘b’ as well as linear outputs, Z) with respect to the cost/loss function. As the name connotes, this chain of sequences computes the partial derivatives, using caches that were stored during the forward propagation sequence, by going ‘backwards’ from the output layer to the input layer. For brevity’s sake, here are the vectorized formulas needed to implement backpropagation in our network.
"""

# ╔═╡ b9547c27-a861-45b9-8877-b222e6f5b5e7
md"""
**Backpropogation**
$(LocalResource("D:\\Materials_and_Documents\\Research_and_Publication\\Programming\\Julia\\tutorial\\Neural_Network\\images\\backpropogation_vectorized.png"))
"""

# ╔═╡ 058984ac-8f9e-4ac3-b821-dfa94390714f
md"""
These two utility functions implements the derivatives of our `sigmoid` and `ReLu` activations:
"""

# ╔═╡ 3eb192c7-4419-4884-babf-616d7bb45757
begin
"""
    Derivative of the Sigmoid function.
"""
function sigmoid_backwards(∂A, activated_cache)
    s = sigmoid(activated_cache).A
    ∂Z = ∂A .* s .* (1 .- s)

    @assert (size(∂Z) == size(activated_cache))
    return ∂Z
end


"""
    Derivative of the ReLU function.
"""
function relu_backwards(∂A, activated_cache)
    return ∂A .* (activated_cache .> 0)
end
end

# ╔═╡ a9bd4097-e323-448d-b8fb-cf7bf8ad688f
md"""
With the derivative of the activation functions, let’s focus on unpacking the stored components of a linear activated output (weight, bias, & activated output of the previous layer) and computing the partial derivatives of each of them with respect to the loss function with these two functions.
"""

# ╔═╡ 6edc0cc9-340d-4665-98e2-5298ae70ad95
begin
"""
    Partial derivatives of the components of linear forward function
    using the linear output (∂Z) and caches of these components (cache).
"""
function linear_backward(∂Z, cache)
    # Unpack cache
    A_prev, W, b = cache
    m = size(A_prev, 2)

    # Partial derivates of each of the components
    ∂W = ∂Z * (A_prev') / m
    ∂b = sum(∂Z, dims = 2) / m
    ∂A_prev = (W') * ∂Z

    @assert (size(∂A_prev) == size(A_prev))
    @assert (size(∂W) == size(W))
    @assert (size(∂b) == size(b))

    return ∂W, ∂b, ∂A_prev
end


"""
    Unpack the linear activated caches (cache) and compute their derivatives
    from the applied activation function.
"""
function linear_activation_backward(∂A, cache, activation_function="relu")
    @assert activation_function ∈ ("sigmoid", "relu")

    linear_cache , cache_activation = cache

    if (activation_function == "relu")

        ∂Z = relu_backwards(∂A , cache_activation)
        ∂W, ∂b, ∂A_prev = linear_backward(∂Z , linear_cache)

    elseif (activation_function == "sigmoid")

        ∂Z = sigmoid_backwards(∂A , cache_activation)
        ∂W, ∂b, ∂A_prev = linear_backward(∂Z , linear_cache)

    end

    return ∂W, ∂b, ∂A_prev
end
end

# ╔═╡ d2911f81-5cba-400b-9a4e-7b574563af3c
md"""
To reconstruct the original `αⱼ`s just multiply `zⱼ[idx] .* τ`:

```julia
τ = summarystats(chain_ncp)[:τ, :mean]
αⱼ = mapslices(x -> x * τ, chain_ncp[:,namesingroup(chain_ncp, :zⱼ),:].value.data, dims=[2])
chain_ncp_reconstructed = hcat(Chains(αⱼ, ["αⱼ[$i]" for i in 1:length(unique(idx))]), chain_ncp)
```
"""

# ╔═╡ 7fc9e1cd-17fc-4db3-900b-0aa3d90239e2
md"""
Combining the derivate of the activation functions and the stored components of each layers activated outputs, backpropagation algorithm is composed for the network with this function.
"""

# ╔═╡ f24eb7b4-82ef-49dd-ab88-1f088ae932ec
begin
"""
    Compute the gradients (∇) of the parameters (master_cache) of the constructed model with respect to the cost of predictions (Ŷ) in comparison with actual output (Y).
"""
function back_propagate_model_weights(Ŷ, Y, master_cache)
    # Initiate the dictionary to store the gradients for all the components in each layer
    ∇ = Dict()

    L = length(master_cache)
    Y = reshape(Y , size(Ŷ))

    # Partial derivative of the output layer
    ∂Ŷ = (-(Y ./ Ŷ) .+ ((1 .- Y) ./ ( 1 .- Ŷ)))
    current_cache = master_cache[L]

    # Backpropagate on the layer preceeding the output layer
    ∇[string("∂W_", (L))], ∇[string("∂b_", (L))], ∇[string("∂A_", (L-1))] = linear_activation_backward(∂Ŷ,
                                                                                                       current_cache,
                                                                                                       "sigmoid")
    # Go backwards in the layers and compute the partial derivates of each component.
    for l=reverse(0:L-2)
        current_cache = master_cache[l+1]
        ∇[string("∂W_", (l+1))], ∇[string("∂b_", (l+1))], ∇[string("∂A_", (l))] = linear_activation_backward(∇[string("∂A_", (l+1))],
                                                                                                             current_cache,
                                                                                                             "relu")
    end

    # Return the gradients of the network
    return ∇
end
end

# ╔═╡ 6251e744-bda3-4285-8dcf-008b21a756ac
md"""
## Optimization Technique
"""

# ╔═╡ a14298a9-9d56-4729-a57c-526ae83b508d
md"""
We are almost there! Our network is taking shape now. When given some layers as an architecture, it can initiate the parameters/weights of these layers randomly, generate some random predictions thereof and quantify how right or wrong overall it is from the ground truth using this function;
"""

# ╔═╡ f719e931-3008-4517-82d3-10d401a601b6
begin
"""
    Check the accuracy between predicted values (Ŷ) and the true values(Y).
"""
function assess_accuracy(Ŷ , Y)
    @assert size(Ŷ) == size(Y)
    return sum((Ŷ .> 0.5) .== Y) / length(Y)
end
end

# ╔═╡ 4d1b731e-0f8b-4804-8d79-500d1d54c404
md"""
Additionally, backpropagation steps now allows the network to measure how each of these weights changes with respect to loss function. Thus, all the networks needs to do is to tweak these different weights such that its predictions match as much as possible with the actual outputs of the data given to this network.

In theory, one could try all possible ranges of all the parameters in the network and select the network that gives the best possible loss (as determined by the output value of the loss function). In practice, this inefficient approach is impractical for the simple fact that it would be computationally too expensive for finding the best parameters of a network with so many parameters. For this and other reasons, an optimization technique is used to find the best parameters. Gradient Descent is one of the most popular optimization techniques for find the best parameters of such a network. Let’s now try to explain this optimization technique with a simple practical analogy.

Suppose you are placed on the top mountain Everest given a challenge of finding the lowest part of the moutain (where a bag full of gold has been placed as the price) while wearing a blindfold. The only help one can get is that at every point in this journey, he/she can talk to a guide via a walkie talkie and ONLY ask about the current direction relative to the highest point of the mountain (gradient) at that point. Gradient descent, as an optimization technique, proposes a simple but effective way of helping one to navigate this challenge (given the constraints). A potential solution offered by this technique is that, one starts at some point and at different steps uses the walkie-talkie to get the relative direction to the highest point of Everest and then move some steps (learning rate) in the opposite direction given. Repeat these technique until one gets to the bag of gold at the lowest part of Everest.
"""

# ╔═╡ 936ce363-c10e-45fb-a9ca-a672be7a714c
md"""
**Backpropogation**
$(LocalResource("D:\\Materials_and_Documents\\Research_and_Publication\\Programming\\Julia\\tutorial\\Neural_Network\\images\\grad_descent1.png"))
"""

# ╔═╡ f777f53a-87a9-4202-8311-de7d2373b3f5
md"""
This snippet applies Gradient descent in our network to update the parameters of the network.
"""

# ╔═╡ ab061851-2091-453f-8039-3deab4791eaa
begin
"""
    Update the paramaters of the model using the gradients (∇)
    and the learning rate (η).
"""
function update_model_weights(parameters, ∇, η)
    L = Int(length(parameters) / 2)

    # Update the parameters (weights and biases) for all the layers
    for l = 0: (L-1)
        parameters[string("W_", (l + 1))] -= η .* ∇[string("∂W_", (l + 1))]
        parameters[string("b_", (l + 1))] -= η .* ∇[string("∂b_", (l + 1))]
    end
    return parameters
end
end

# ╔═╡ 0083b86d-205e-4d51-826e-94e3c97a49e2
md"""
## Training the Network 
"""

# ╔═╡ 174fe26d-4dd1-493e-9927-03a4c343bdfe
md"""
The training of a neural neural is basically the chain of sequence where input data are forwarded through the network using available parameters of this network, predictions compared to the actual training data predictions and then finally the tweaking of the parameters as a method of improving predictions. Each run of this sequence is referred to as an ‘epoch’ in machine learning lingo. The core idea of this training phase is to try to improve predictions generated by the network at various attempts or iterations (epoch). This snippet combines all the various components that we have generated so far to do just that:
"""

# ╔═╡ e09f0384-4648-4183-9fad-acc959ff5fad
begin
"""
    Train the network using the desired architecture that best possible
    matches the training inputs (DMatrix) and their corresponding ouptuts(Y)
    over some number of iterations (epochs) and a learning rate (η).
"""
function train_network(layer_dims , DMatrix, Y;  η=0.001, epochs=1000, seed=2020, verbose=true)
    # Initiate an empty container for cost, iterations, and accuracy at each iteration
    costs = []
    iters = []
    accuracy = []

    # Initialise random weights for the network
    params = initialise_model_weights(layer_dims, seed)

    # Train the network
    for i = 1:epochs

        Ŷ , caches  = forward_propagate_model_weights(DMatrix, params)
        cost = calculate_cost(Ŷ, Y)
        acc = assess_accuracy(Ŷ, Y)
        ∇  = back_propagate_model_weights(Ŷ, Y, caches)
        params = update_model_weights(params, ∇, η)

        if verbose
            println("Iteration -> $i, Cost -> $cost, Accuracy -> $acc")
        end

        # Update containers for cost, iterations, and accuracy at the current iteration (epoch)
        push!(iters , i)
        push!(costs , cost)
        push!(accuracy , acc)
    end
        return (cost = costs, iterations = iters, accuracy = accuracy, parameters = params)
end
end

# ╔═╡ 3aa75aca-5886-4de7-bfb0-05bf07b675b1
begin

using MLJBase
# Generate fake data
X, y = make_blobs(10_000, 3; centers=2, as_table=false, rng=2020);
X = Matrix(X');
y = reshape(y, (1, size(X, 2)));
f(x) =  x == 2 ? 0 : x
y2 = f.(y); 
 
# Input dimensions 
input_dim = size(X, 1);

# Train the model
nn_results = train_network([input_dim, 5, 3, 1], X, y2; η=0.01, epochs=50, seed=1, verbose=true);

# Plot accuracy per iteration
p1 = plot(nn_results.accuracy,
         label="Accuracy",
         xlabel="Number of iterations",
         ylabel="Accuracy as %",
         title="Development of accuracy at each iteration");

# Plot cost per iteration 
p2 = plot(nn_results.cost,
         label="Cost",
         xlabel="Number of iterations",
         ylabel="Cost (J)",
         color="red",
         title="Development of cost at each iteration");

# Combine accuracy and cost plots
plot(p1, p2, layout = (2, 1), size = (800, 600))
end

# ╔═╡ 4fe36ee6-f13c-4001-aaf3-010714bbc384
md"""
With all the components in place, it is time to test if the implementation works. For the sake of brevity, let us just generate some simple synthetic binary classification data for this demonstration. This is easily done as:
"""

# ╔═╡ 808a77c1-3c5d-4950-9bda-e975a234c75a
md"""
## Conclusion
"""

# ╔═╡ 4ad1654c-e2e6-44a9-88a5-ff138d04c72f
md"""
As the plots show the network starts with a low accuracy and high cost values but as it keeps learning and updating the parameters of the network, accuracy rises (while cost plateaus) until it achieves a perfect score on this super simple and perfect dummy dataset. Sadly, datasets are rarely this perfect in real life!
"""

# ╔═╡ 2c3773c9-d9d3-40b2-b694-41dce4214acc
md"""
### Pluto Stuff
"""

# ╔═╡ 3da23693-99be-4b28-a869-cc84b16d3992
TableOfContents(title="Artificial Neural Network", aside=true)    

# ╔═╡ f9545591-fcb7-4d6a-8e16-9cfdf5eaab2e
md"""
### Environment
"""

# ╔═╡ e40f1b36-4701-4e07-bb34-580c90824bd3
with_terminal() do
	deps = [pair.second for pair in Pkg.dependencies()]
	deps = filter(p -> p.is_direct_dep, deps)
	deps = filter(p -> !isnothing(p.version), deps)
	list = ["$(p.name) $(p.version)" for p in deps]
	sort!(list)
	println(join(list, '\n'))
end

# ╔═╡ Cell order:
# ╟─bd97eae6-e049-4404-929d-5505e46b3b69
# ╟─023b3c90-03a0-11ec-3ec9-b5090a456e13
# ╟─f01ef8f2-1032-443c-9782-87d815ce3d2b
# ╟─bb3cd33f-7bb4-47a2-ba88-5f4c60e69b39
# ╟─abab9e4f-0d1e-40f7-ab19-87701d767b78
# ╟─e8aa1fc8-3394-4c66-a641-607d1e38a88c
# ╟─61c9d2d8-42c9-4d35-b9d6-73e8b5fab586
# ╟─e83cedb6-bc03-467b-bb4f-97f64e353769
# ╟─6fa60d30-dbe7-4bc1-bf0f-fb5a8a809daa
# ╟─3106fec3-e5a5-4ac1-af88-fe91e23574b5
# ╟─d090a450-5cdf-4b89-9e2e-5709613f4724
# ╟─64aa304b-fc5e-4033-967a-7aea84c9082d
# ╟─a9d7f452-3df6-4834-9764-039d85531c50
# ╟─8a60aeb1-45a1-4203-ae2c-782bd1032416
# ╟─e01d8ed6-5124-4be9-946e-29bb587ebb72
# ╟─0ad489ba-b9b8-4793-86bc-17267cd387f9
# ╟─ddae3887-a59e-4dd3-94ca-47f90f46623b
# ╟─db370532-3b03-40d7-a07a-732d70545372
# ╟─d07c30a6-8acf-4dd5-9f94-38730a8be4b2
# ╟─b4c13e29-7e77-4a25-936d-542641e89e52
# ╠═0ca007a0-c44b-4858-af07-7abec19bfaed
# ╟─c4dffeb7-2dc4-43ed-9cca-040e791fbd37
# ╟─df589864-1d1a-4565-a3a7-d0537d8d9d5e
# ╟─ef9b1adb-ba9a-49f7-b840-3f33cbfeefab
# ╟─547ef676-7baf-4685-bbce-3bb649310274
# ╠═3c07574a-624a-4d58-918d-99f5d8f73dc0
# ╟─00c74375-bcb6-4ec8-a489-3f62e9b0b682
# ╟─a7d21076-9c5b-4147-ad4a-1bec8a5e7954
# ╟─fefa0375-76ef-4dd4-81d1-9ddb31c67274
# ╟─05a2d223-737d-4646-9a45-1e062144f923
# ╠═c21216e9-c8dc-48a7-9183-0651a48c79e2
# ╟─443476ef-5878-45fa-bc65-60e2066b18a0
# ╠═3cdd0065-5545-4d87-b7ba-973ac29bfed2
# ╟─1564c00b-48cc-484c-bc3c-afb47f33404f
# ╟─989c7a4b-5332-4192-96d1-b00d44f54242
# ╟─8791f35a-d73b-43dd-aa95-2a35d41308ad
# ╠═e39a4243-0042-44db-a359-b51e4f46d162
# ╟─ad76b1bc-f4b0-4400-a175-e5d0d8cf2fdf
# ╟─e3896c90-cebc-44c0-b6b5-6a0114cf1744
# ╟─b9547c27-a861-45b9-8877-b222e6f5b5e7
# ╟─058984ac-8f9e-4ac3-b821-dfa94390714f
# ╠═3eb192c7-4419-4884-babf-616d7bb45757
# ╟─a9bd4097-e323-448d-b8fb-cf7bf8ad688f
# ╠═6edc0cc9-340d-4665-98e2-5298ae70ad95
# ╟─d2911f81-5cba-400b-9a4e-7b574563af3c
# ╟─7fc9e1cd-17fc-4db3-900b-0aa3d90239e2
# ╠═f24eb7b4-82ef-49dd-ab88-1f088ae932ec
# ╟─6251e744-bda3-4285-8dcf-008b21a756ac
# ╟─a14298a9-9d56-4729-a57c-526ae83b508d
# ╠═f719e931-3008-4517-82d3-10d401a601b6
# ╟─4d1b731e-0f8b-4804-8d79-500d1d54c404
# ╟─936ce363-c10e-45fb-a9ca-a672be7a714c
# ╟─f777f53a-87a9-4202-8311-de7d2373b3f5
# ╠═ab061851-2091-453f-8039-3deab4791eaa
# ╟─0083b86d-205e-4d51-826e-94e3c97a49e2
# ╟─174fe26d-4dd1-493e-9927-03a4c343bdfe
# ╠═e09f0384-4648-4183-9fad-acc959ff5fad
# ╟─4fe36ee6-f13c-4001-aaf3-010714bbc384
# ╠═3aa75aca-5886-4de7-bfb0-05bf07b675b1
# ╟─808a77c1-3c5d-4950-9bda-e975a234c75a
# ╟─4ad1654c-e2e6-44a9-88a5-ff138d04c72f
# ╟─2c3773c9-d9d3-40b2-b694-41dce4214acc
# ╠═3da23693-99be-4b28-a869-cc84b16d3992
# ╠═b8b8f9d2-0f09-4c82-aeb9-33b1ca0cc71e
# ╟─f9545591-fcb7-4d6a-8e16-9cfdf5eaab2e
# ╟─5fda3c7f-2045-41ea-af20-ef2a987fa0c7
# ╟─e40f1b36-4701-4e07-bb34-580c90824bd3
