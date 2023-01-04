1. RNNS
    1. [E] What’s the motivation for RNN?
        The motivation behind Recurrent Neural Networks is to capture the dependencies of the observations in a sequence of data, e.g. time series or natural language. For instance, in natural language the words in a sentence are not independent of one another and knowing the first three words can help you guess the forth word. RNNs aim to capture this dependency by passing the information from the previous inputs when processing the current input.
    1. [E] What’s the motivation for LSTM?
        RNNs have the issue of long-term dependencies due to vanishing gradients, LSTMs (and GRUs) were introduced to overcome this issue by allowing information from early layers directly to later layers. The forget gates in LSTMs structure allows for the network to selectively remember/forget information from time step to the next.
    1. [M] How would you do dropouts in an RNN?
        Dropout can be applied in a RNN in different ways:

        1. It can be applied to the hidden state that goes to the output and not to the next timestamp. Note that different samples in a mini-batch should have different dropout masks but the same sample in different time steps should have the same mask
        2. It can be applied to the inputs x_t
        3. It can be applied to the weights between the hidden states. Note that the same dropout mask should be used for all time steps in a mini-batch
2. [E] What’s density estimation? Why do we say a language model is a density estimator?
    Density estimation means estimating the probability density function (PDF) of a random variable from a set of observations. The PDF of a variable describes the probability of the variable taking on different values. 

    Language models are trained on sequences of words to learn the probability of words occurring. In other words, they are estimating the PDF of word sequences and can therefore be interpreted as density estimators.
3. [M] Language models are often referred to as unsupervised learning, but some say its mechanism isn’t that different from supervised learning. What are your thoughts?
    Language models are trained on vast amounts of text without any explicit labels. In that regard they are unsupervised. But in order for the model to learn the intricacies of the language, the relationship between different words it is usually trained in an auto-regressive manner, i.e. a set of words are masked and the model is trained to predict the masked words. These masked words can be thought of as labels which is similar to supervised learning. 