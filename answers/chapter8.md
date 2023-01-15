#### 8.2.1 Natural language processing
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
4. Word embeddings.
    
    1. [M] Why do we need word embeddings?
    
    Word embeddings are a way to map words to vector representations that can be used in matrix multiplication in neural networks. These representations preserve the semantics and are lower in dimension than one-hot encoded vectors.
    
    2. [M] What’s the difference between count-based and prediction-based word embeddings?
    
    Count-based embeddings learn the embeddings based on the co-occurrences of words across a large dataset. GloVe is a count-based embedding method. Prediction-based word embeddings learns the embeddings by learning to predict a word of set of words based on the surrounding words and minimising the prediction loss
    
    3. [H] Most word embedding algorithms are based on the assumption that words that appear in similar contexts have similar meanings. What are some of the problems with context-based word embeddings?
    
    Context-based embeddings reinforce gender and racial biases present in the training data. For example the embedding of the word smart or beautiful should not have any gender preferences baked into it but if you ask a LM to describe someone smart or translate a sentence describing someone smart from a gender-neutral language to English for instance, it will prefer the pronoun he for smart and she for beautiful because in the context of other words in the training data smart is more associated with males and beauty with females. 
    
    Another issue can be that words with different meaning will have different embeddings which make it difficult for them to be used standalone.
5. Given 5 documents:
		D1: The duck loves to eat the worm
		D2: The worm doesn’t like the early bird
		D3: The bird loves to get up early to get the worm
		D4: The bird gets the worm from the early duck
		D5: The duck and the birds are so different from each other but one thing they have in common is that they both get the worm
    1. [M] Given a query Q: “The early bird gets the worm”, find the two top-ranked documents according to the TF/IDF rank using the cosine similarity measure and the term set {bird, duck, worm, early, get, love}.  Are the top-ranked documents relevant to the query?
        Each document/query has to go through lemmatization and stemming. Then, the TF and IDF for each document and query is calculated and multiplied together. Finally the cosine similarity between the query's TF/IDF vector and each documents' determines the rank of the similarities. The top two are selected:
        The IDF for the documents and set above is: 
            {bird: 0.22, duck: 0.51, worm: 0, early: 0.51, get: 0.51, love: 0.92}
        This is because 4 out of 5 documents contain the word "bird" so the natural log of 5 documents over 4 is 0.22 and so on. Below are the term frequencies for the query and documents:
            D1: {bird: 0, duck: 1, worm: 1, early: 0, get: 0, love: 1}
            D2: {bird: 1, duck: 0, worm: 1, early: 1, get: 0, love: 0}
            D3: {bird: 1, duck: 0, worm: 1, early: 1, get: 2, love: 1}
            D4: {bird: 1, duck: 1, worm: 1, early: 1, get: 1, love: 0}
            D5: {bird: 1, duck: 1, worm: 1, early: 0, get: 1, love: 0}
            Q: {bird: 1, duck: 0, worm: 1, early: 1, get: 1, love: 0}
        With this the TF/IDF of each document and query becomes:
            D1: [0, .51, 0, 0, 0, .92]
            D2: [.22, 0, 0, .51, 0, 0]
            D3: [.22, 0, 0, .51, 1.02, .92]
            D4: [.22, .51, 0, .51, .51, 0]
            D5: [.22, .51, 0, 0, .51, 0]
            Q: [.22, 0, 0, .51, .51, 0]
        Now the cosine similarity between the query and each document is:
            cos(Q, D1) = 0
            cos(Q, D2) = .737
            cos(Q, D3) = .742
            cos(Q, D4) = .828
            cos(Q, D5) = .543
        So top 2 documents are D4 and D3. They are relevant in the sense that they share common words. D3 seems to share more similarity in semantics than D4 which mentions the "early duck".
    1. [M] Assume that document D5 goes on to tell more about the duck and the bird and mentions “bird” three times, instead of just once. What happens to the rank of D5? Is this change in the ranking of D5 a desirable property of TF/IDF? Why?
        This changes the TF of D5 to {bird: 3, duck: 1, worm: 1, early: 0, get: 1, love: 0} which results in a TF/IDF of [.66, 0, 0, .51, .51, 0]. This increases the cosine similarity score between the query and D5 to .55 which does not change the overall ranking.
        This change is not a desirable property of TF/IDF because a document can just copy and paste a word hundreds of times and increase its TF/IDF score without adding relevant information.

6. [E] Your client wants you to train a language model on their dataset but their dataset is very small with only about 10,000 tokens. Would you use an n-gram or a neural language model?
    Depends on the specific tasks and constraints. In general, if the dataset is small and the task does not require modeling complex syntactic and semantic relationships, you might consider using a n-gram language model, as it may be more efficient and easier to implement. If the task requires modeling complex syntactic and semantic relationships and the dataset is large enough to support the learning of a neural language model, you might consider using a neural language model.
7. [E] For n-gram language models, does increasing the context length (n) improve the model’s performance? Why or why not?
    To some extent, increasing n from 1 to 3 for example helps with capturing the context better through longer range dependencies. However, increasing n at some point will have a negative affect on the model's generalization ability and computational efficiency. As the context length (n) increases, the number of possible n-grams increases exponentially, which can lead to sparsity in the data and poor generalization to unseen data.
8. [M] What problems might we encounter when using softmax as the last layer for word-level language models? How do we fix it?
    The issue is that with large vocabularies, the softmax computation is very expensive as there are B (batch size) * d (model dimension) * V (vocab size)parameters. 
    There are a number of alternatives to using a standard softmax layer:
        1. Hierarchical softmax: Words are leaves of a tree and nodes and instead of predicting the probability of each word, the probability of nodes are predicted
        1. Differentiated softmax: Is based on the intuition that not all words require the same number of parameters: Many occurrences of frequent words allow us to fit many parameters to them, while extremely rare words might only allow to fit a few
        1. Sampling softmax: By using different sampling techniques, e.g. negative sampling, this alternative approximates the normalization in the denominator of the softmax with some other loss that is cheap to compute. However, sampling-based approaches are only useful at training time -- during inference, the full softmax still needs to be computed to obtain a normalised probability.
    Related articles: https://towardsdatascience.com/how-to-overcome-the-large-vocabulary-bottleneck-using-an-adaptive-softmax-layer-e965a534493d, https://ruder.io/word-embeddings-softmax/index.html#hierarchicalsoftmax
9. [E] What's the Levenshtein distance of the two words “doctor” and “bottle”?
    The distance is 4: Replace "d", "c", "o" and "r"
10. [M] BLEU is a popular metric for machine translation. What are the pros and cons of BLEU?
    Pros:
    1. It is widely used so different models can be compared with one another
    2. It is easy to implement. It only needs the target and prediction to calculate the precision for different n-grams
    Cons:
    1. Does not consider semantics and only relies on same tokens this has two issues: it penalises translations that convey the same meaning but use different words. On the other hand, it doesn’t penalise translations that are semantically incorrect but have a lot of overlapping words
11. [H] On the same test set, LM model A has a character-level entropy of 2 while LM model A has a word-level entropy of 6. Which model would you choose to deploy?
    The entropy is just a measure of randomness in the model and having a lower entropy doesn’t mean the model is better. The models should be compared on metrics useful for the task the model is going to be used for in production.
12. [M] Imagine you have to train a NER model on the text corpus A. Would you make A case-sensitive or case-insensitive?
    Depends on the target entities and the data. If we want the model to distinguish between say Apple the company and apple the fruit, should consider case-sensitivity. However, if the data contains a mix of upper and lower case of the same words of the same entity, enforcing case sensitivity can be confusing to the model.
13. [M] Why does removing stop words sometimes hurt a sentiment analysis model?
    Because the removal of some stopwords such as negating words (no, not, etc.), change the semantics. For example, a negative review that says: "Do not buy this product! It is no good" will turn into "Do buy this product! It is good" after removing stopwords which has the exact opposite meaning of the original review.
14. [M] Many models use relative position embedding instead of absolute position embedding. Why is that?
    Relative position embedding can generalize to unknown sequence lengths because it encodes the distance between tokens whereas absolute position embeddings is limited to a fixed length. 
15. [H] Some NLP models use the same weights for both the embedding layer and the layer just before softmax. What’s the purpose of this?
    From: https://paperswithcode.com/method/weight-tying#:~:text=Weight%20Tying%20improves%20the%20performance,that%20it%20is%20applied%20to. 
    Weight Tying improves the performance of language models by tying (sharing) the weights of the embedding and softmax layers. This method also massively reduces the total number of parameters in the language models that it is applied to.
    Language models are typically comprised of an embedding layer, followed by a number of Transformer or LSTM layers, which are finally followed by a softmax layer. Embedding layers learn word representations, such that similar words (in meaning) are represented by vectors that are near each other (in cosine distance). [Press & Wolf, 2016] showed that the softmax matrix, in which every word also has a vector representation, also exhibits this property. This leads them to propose to share the softmax and embedding matrices, which is done today in nearly all language models.
#### 8.2.2 Computer vision
1. [M] For neural networks that work with images like VGG-19, InceptionNet, you often see a visualization of what type of features each filter captures. How are these visualizations created?
    One common approach for creating these visualizations is to use an algorithm called "feature maximization". This algorithm starts with a random image and then repeatedly applies a small change to the image that maximizes the activation of a particular filter. Through this process, an image is generated that is specifically tailored to activate that filter.

    Another approach is to use "saliency maps". In this approach, an image is fed into the neural network, and the network produces activations for each filter. Next, the gradient of the output of a particular filter with respect to the input image is calculated. This gradient can then be used to create a heatmap where the brightness represents the importance of each pixel in the input image for that filter.
1. Filter size.
    1. [M] How are your model’s accuracy and computational efficiency affected when you decrease or increase its filter size?
        For some tasks such as object detection, bigger filter sizes are better as they capture more contextual information and the relationship between different parts of the image. However, for segmentation tasks the local features are more important and therefore a smaller filter size may result in higher accuracy.
        In regards to computational efficiency, the bigger the filter size, the more parameters the model needs to learn and therefore the more computation and memory it requires.
    1. [E] How do you choose the ideal filter size?
        It is common to experiment with different filter sizes and evaluate the model's performance using different evaluation metrics, this will give an idea of which filter size works best on the specific task and data set. If the task is object detection, a bigger kernal size may be better since the context and relationship between different parts of the image is important. However, if the task is segmentation, smaller sizes are better to preserve spatial information. Another thing to keep in mind when deciding on the filter sizes is the computational complexity of larger kernel sizes as they introduce more parameters. Generally, it is common to have smaller kernel sizes in the initial layers where local features are extracted and increase the size in deeper layers to extract more abstract features.
1. [M] Convolutional layers are also known as “locally connected.” Explain what it means.
    The term "locally connected" refers to the fact that the neurons in a convolutional layer are connected only to a small region of the input image, rather than to the entire image. Each neuron in a convolutional layer is connected to a small subset of the input image, and these subsets are called "receptive fields". These receptive fields are of the same size and arranged in the same way as the kernel of the convolutional layer and slide over the input image in a process called convolution.
    A key feature of locally connected layers is that they are able to extract spatial features in the input data that are translation-invariant, meaning they can identify objects and patterns regardless of their location in the image.
    For example, consider an image of a face, in which the face can appear at different positions in the image. If we use a fully connected layer, the weight of the neurons would have to be adjusted for all possible positions of the face. But by using locally connected layers, the model only needs to learn the features of the face, regardless of its location, making it less computationally expensive and more robust to changes in position.
1. [M] When we use CNNs for text data, what would the number of channels be for the first conv layer?
    Similar to grayscale data, one channel is used for text data
1. [E] What is the role of zero padding?
    Zero padding is the process of adding zeros to the edges of the input. One reason for this is to enforce a certain size to the output of the convolution. Another benefit of zero padding is that more edge pixels will be included in the convolution and therefore more information will be captured.
1. [E] Why do we need upsampling? How to do it?
    Upsampling is needed to restore the desired resolution after downsampling. There are different technicques for upsampling, some are independent of the input data. For example, Nearest Neighbors, Interpolation or Bed of Nails. All these methods involve copying some of the input values or filling in zeros in some postions. Another technique is called Transposed Convolutions which involves striding a kernal on the downsampled image. To elaborate, each element in the input is multiplied with each element in the kernal and the overlapping results are summed up. The striding kernal is learned during training so unlike the other techniques it is dependant on the data. 
    Here's an article with illustrations of these techniques: https://towardsdatascience.com/transposed-convolution-demystified-84ca81b4baba  
1. [M] What does a 1x1 convolutional layer do?
    It's used as a dimensionality reduction method to reduce the number of feature maps before applying expensive convolutions in the further layers.
1. Pooling.
    1. [E] What happens when you use max-pooling instead of average pooling?
        All the features from the filter are considered and passed to the next layer. This results in a smoother image compared to the output of max pooling which detects the sharp and brighter pixels.
    1. [E] When should we use one instead of the other?
        If the input images are objects with a dark background and we are interested in detecting the foreground objects, max-pooling is better
    1. [E] What happens when pooling is removed completely?
        It depends on the task and objective. Average pooling will include all the features in the feature map whereas max pooling has data loss and only considers the highest values and misses out on the other details related to the rest of the image. If the task is to detect edges for example, max pooling is a better choice. 
    1. [M] What happens if we replace a 2 x 2 max pool layer with a conv layer of stride 2?
        Replacing a 2x2 max pool layer with a convolutional layer with a stride of 2 would result in the same spatial downsampling of the feature maps. However, the main difference is that a convolutional layer also learns to extract features from the input data, while a max pooling layer only performs spatial downsampling. Also, the conv layer adds to the number of learnable parameters while max pooling doesn't.
1. [M] When we replace a normal convolutional layer with a depthwise separable convolutional layer, the number of parameters can go down. How does this happen? Give an example to illustrate this.
    A good article with example and illutstrations can be found here: https://www.geeksforgeeks.org/depth-wise-separable-convolutional-neural-networks/
1. [M] Can you use a base model trained on ImageNet (image size 256 x 256) for an object classification task on images of size 320 x 360? How?
    The different image sizes may result in a different context or scale, which could affect the model's performance. Also, the model trained on a different input size may not be familiar with the different aspect ratios from a different input size. That being said, you can transform your inputs to match the expected 256 x 256 size. Alternatively, you can finetune ImageNet on images with size 320 x 360.
1. [H] How can a fully-connected layer be converted to a convolutional layer?
    The neurons need to be reshaped to have two dimensions, the weights need to be rearranged to match the desired kernal size and number of filters. A good explanation with illustration can be found here: https://sebastianraschka.com/faq/docs/fc-to-conv.html#:~:text=There%20are%20two%20ways%20to,1x1%20convolutions%20with%20multiple%20channels.
1. [H] Pros and cons of FFT-based convolution and Winograd-based convolution.
    FFT-based convolution:
    Pros:
    1. Speed up
    1. It can be used to implement convolution operations of any kernel size
    Cons:
    1. It requires a large amount of memory to store the transformed data, which can be a problem for large-scale neural networks.
    1. It can be sensitive to numerical errors and rounding issues, which could affect the accuracy of the results.
    Winograd-based convolution:
    Pros:
    1. Speed up
    Cons:
    1. It is not suitable for large kernel sizes and it's not as general as the FFT-based convolution.
    1. It requires a large amount of memory to store the transformed data, which can be a problem for large-scale neural networks.
#### 8.2.3 Reinforcement learning

> 🌳 **Tip** 🌳<br>
To refresh your knowledge on deep RL, checkout [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/) (OpenAI)


28. [E] Explain the explore vs exploit tradeoff with examples.
29. [E] How would a finite or infinite horizon affect our algorithms?
30. [E] Why do we need the discount term for objective functions?
31. [E] Fill in the empty circles using the minimax algorithm.

	<center>
		<img src="images/image34.png" width="45%" alt="Minimax algorithm" title="image_tooltip">
	</center>

32. [M] Fill in the alpha and beta values as you traverse the minimax tree from left to right.

    <center>
		<img src="images/image35.png" width="80%" alt="Alpha-beta pruning" title="image_tooltip">
	</center>

33. [E] Given a policy, derive the reward function.
34. [M] Pros and cons of on-policy vs. off-policy.
35. [M] What’s the difference between model-based and model-free? Which one is more data-efficient?

#### 8.2.4 Other

36. [M] An autoencoder is a neural network that learns to copy its input to its output. When would this be useful?
        The main goal of an autoencoder is to learn the latent represenation of the input, such that the output can reconstruct the input from this compact, low dimensional latent space. 
        There are many use cases for this:
            1. Compression and storage: autoencoders can be used to reduce the size of the input by learning a compact representation and reconstructing it back when needed. 
            2. Denoising: the encoder portion of an autoencoder can be used to remove the noise from the input while preserving the underlying structure and outputting the denoised version from the decoder.
            3. Anomoly detection: The learned representations of the input can be used to identify outliers at inference.
            4. Generative models: The learned latent space of the encoder can be sampled and used by the generator to generate new data similar to the inputs.
            5. Transfer learning and feature extraction: An encoder from the pretrained autoencoder can be used as an embedder to project the inputs to an embedding space which can be used as features to a classification model for example.
37. Self-attention.
    15. [E] What’s the motivation for self-attention?
            The motivation behind self-attention is for an the model to attend to different parts of the inputs that are relevant to the task at hand. It does this by calculating weights for each of the parts. The parts that are more relevant to the task at hand get higher weights. These weights are used to determine the contribution of different input componenets when making a predition.
    16. [E] Why would you choose a self-attention architecture over RNNs or CNNs?
            1. One limitation of RNNs and CNNs that self-attention resolves is assigning different weights to the inputs based on their relevance.
            2. Attention-based models are better with longer-term dependecies
            3. Attention-based models can run in parallel and are therefore more computationally efficient than RNNs
    17. [M] Why would you need multi-headed attention instead of just one head for attention?
        According to the Attention Is All You Need paper (https://arxiv.org/pdf/1706.03762.pdf), multi-head attention allows the model to attend to words other than the current input from different representation subspaces. In other words, using multiple heads in the allows the model to learn different types of relationships between elements in the input sequence and attend to different granularity and modalities of the input, which improves the performance of the model. 
    18. [M] How would changing the number of heads in multi-headed attention affect the model’s performance?
        Depending on the amount of data and the task complexity, increasing the number of heads may improve the model's performance as more heads allows for different types of relationships between the input elements to be learned. Increasing it too much may not be useful as the model may not learn any new representation subspaces. The number is a hyper-parameter that needs to be tuned like any other hyper-parameter.
38. Transfer learning
    19. [E] You want to build a classifier to predict sentiment in tweets but you have very little labeled data (say 1000). What do you do?
        There are a number of ways:
            1. Create more data through data augmentation technniques such as back-translation, synonym replacement, random replacement, etc.
            2. Fine-tuning a pre-trained model that has been trained for sentiment classification
    20. [M] What’s gradual unfreezing? How might it help with transfer learning?
        Gradual unfreezing is a way to fine-tune a pretrained model. It's the process of training some of the layers of a pretrained model on a new task or same task with different data at a time and increasing the number of trainable layers over time. This helps with transfer learning because it allows the pretrained model to slowly adjust its weights for the task at hand while maintaining some of the learned information from pretraining. 
39. Bayesian methods.
    21. [M] How do Bayesian methods differ from the mainstream deep learning approach?
        The main difference is that Bayesian methods model a probability distribution of the parameters, and therefore model the uncertainty of predictions whereas mainstream deep learning approaches optimize for reducing the training/validation error and do not model uncertainty.
    22. [M] How are the pros and cons of Bayesian neural networks compared to the mainstream neural networks?
            1. BNNs give uncertainty of predictions
            2. BNNs are more interpretable than mainstream NNs
            3. BNNs less prone to overfitting
            However,
            4. BNNs are computationally expensive during inference, since multiple forward passes is needed
            5. BNNs require more data than mainstream NNs to model the posteriors
    23. [M] Why do we say that Bayesian neural networks are natural ensembles?
        The model predictions are made by averaging over multiple models, each corresponding to a different set of parameter values drawn from the distributions. This process can be seen as a form of model averaging, where the final prediction is the average of the predictions made by multiple models. This can be seen as "natural ensembles", where the different models correspond to different sets of parameter values. The averaging process allows the model to make probabilistic predictions, and to quantify the uncertainty of its predictions.
40. GANs.
    24. [E] What do GANs converge to?
        GANs converge to a Nash equilibrium which is a stable state where neither the generator or discriminator can make furthur improvements by solely changing their strategy. In this state, the generator generates plausible samples and the discriminator is not able to distinguish whether the sample is real or generated. This is known as a "zero-sum" game.
    25. [M] Why are GANs so hard to train?
        The generator and discriminator can get stuck in local minima and get in an infinite loop of making small improvements. In this case, the generator creats images that are slightly more realistic but not representative of the overall real data distribution and the discriminator learns to adapt to these samples and gets better and identifying them. This phenomenon is know as "model collapse".
        There is also the chance of instability where the generator does not generate realistic images and the discriminator is able to identify them as fake.

### 8.3 Training neural networks
41. [E] When building a neural network, should you overfit or underfit it first?
    Better to start simple and gradually add complexity. This way it easier to verify that things are working as expected
42. [E] Write the vanilla gradient update.
    θ = θ - α * ∇θL(θ)
    Where:
        1. θ is the set of parameters of the network.
        2. α is the learning rate, a scalar value that controls the step size of the update.
        3. L(θ) is the loss function, which measures the difference between the predicted output and the true output.
        4. ∇θL(θ) is the gradient of the loss function with respect to the parameters, which represents the direction of the steepest descent in the parameter space.
43. Neural network in simple Numpy.
    26. [E] Write in plain NumPy the forward and backward pass for a two-layer feed-forward neural network with a ReLU layer in between.
        https://github.com/MaryFllh/ml_algorithms/tree/main/neural_net
    27. [M] Implement vanilla dropout for the forward and backward pass in NumPy.
        https://github.com/MaryFllh/ml_algorithms/tree/main/neural_net

44. Activation functions.
    28. [E] Draw the graphs for sigmoid, tanh, ReLU, and leaky ReLU.
    29. [E] Pros and cons of each activation function.
        Sigmoid:
            Pros: Easy to compute, differentiable
            Cons: Susceptible to vanishing gradients. Gradient values are only significant between -3 and 3 and anything outside of that range is close to zero.
        tanh:
            Pros: Easy to compute, differentiable and symetric with a mean of 0
            Cons: Same issue with Sigmoid
        ReLU:
            Pros: Easy to compute and more computationally efficient than sigmoid or tanh. Doesn't have the saturating property of tanh and sigmoid and converges faster
            Cons: Dead neurons
        Leaky ReLU:
            Pros: Solves the dead neurons problem, easy to compute 
            Cons: Gradient of negative values too small can affect training time. Also choosing an appropriate value for the leakage factor α can be tricky, and it's often chosen through trial and error.
    30. [E] Is ReLU differentiable? What to do when it’s not differentiable?
        It is not differentiable at x = 0. However, it is safe to consider the derivative at this point 0 because:
            1. The exact point at which the function is not differentiable is seldom reached in an algorithm.
            2. At the point of non-differentiability, you can assign the derivative of the function at the point “right next” to the singularity and the algorithm will work fine. For example, in ReLU we can give the derivative of the function at zero as 0. It would not make any difference in the backpropagation algorithm because the distance between the point zero and the “next” one is zero.
    31. [M] Derive derivatives for sigmoid function $$\sigma(x)$$ when $$x$$ is a vector.
            y'(x) = sigma(x) * (1 - sigma(x)) , where the function and the subtraction are applied component-wise.
45. [E] What’s the motivation for skip connection in neural works?
    The motivation is to address the problem of vanishing gradients. Skip connections help to address this problem by allowing the gradients to bypass one or more layers in the network and flow directly to the earlier layers. This helps to ensure that the gradients are larger and can more easily flow back through the network, allowing the network to learn more effectively.
46. Vanishing and exploding gradients.
    32. [E] How do we know that gradients are exploding? How do we prevent it?
        One way to know if the gradients are exploding is to monitor the gradients during training and check if the norm of the gradients (i.e. the magnitude of the gradients) is becoming very large. This can be done by printing the norm of the gradients or by using a tool such as TensorBoard to visualize the gradients.
        Another indication of exploding gradients is that the loss may become NaN (Not a Number) or inf (infinity), this happens when the gradients grow to large numbers that can't be handled by the computer.
        There are various ways to prevent exploding gradiants:
            Gradient Clipping: This method involves clipping the gradients to a maximum value, this ensures that the gradients do not become too large.

            Weight Initialization: Choosing appropriate weight initialization methods can also help to prevent gradients from exploding. For example, using techniques such as Glorot initialization or He initialization can help to ensure that the weights of the network are initialized to appropriate values.

            Normalization: Use of techniques such as batch normalization can also help to prevent gradients from exploding by normalizing the inputs to the activation functions, which makes the training process more stable.

            Regularization: Using regularization techniques such as L1 or L2 can also prevent gradients from exploding by adding a penalty term to the loss function that discourages large weights.
    33. [E] Why are RNNs especially susceptible to vanishing and exploding gradients?
        RNNs are particularly susceptible to the vanishing and exploding gradients problem because the gradients can flow through multiple time steps, and the weights are multiplied many times over the time steps. This can cause the gradients to either become very small (vanishing gradients) or very large (exploding gradients) as they flow through the network.
47. [M] Weight normalization separates a weight vector’s norm from its gradient. How would it help with training?
    The basic idea behind weight normalization is to normalize the weights of a network so that they have a fixed norm (magnitude) during training. This is done by dividing the weights by their norm so that the magnitude of the weights is fixed, regardless of the values of the gradients.
    By normalizing the weights in this way, the gradients only need to update the direction of the weights, rather than both the direction and the scale. This can help to make the training process more stable because the gradients only need to adjust the direction of the weights, rather than both the direction and the scale.
    Additionally, weight normalization can help to improve the generalization of the model, as it makes the model less sensitive to the scale of the weights.
48. [M] When training a large neural network, say a language model with a billion parameters, you evaluate your model on a validation set at the end of every epoch. You realize that your validation loss is often lower than your train loss. What might be happening?
    One reason could be that the train loss is calculated at the end of each batch backporpagation. With a large model, the initial batches will be far from the optimal solution and the loss will be high. However, the validation loss is calculated at the end of each epoch which is after all batches have completed their forward and backward passes and made multiple updates to the weights. In other words, by then end of an epoch the the model has learned more and become more stable so the validation loss can be lower.
49. [E] What criteria would you use for early stopping?
    Validation accuracy or loss. When the validation accuracy (loss) starts to decease (increase) or becomes stagnant, it is an indication that the model is overfitting, and the training process should be stopped.
50. [E] Gradient descent vs SGD vs mini-batch SGD.
    Gradient Descent: The classic gradient descent algorithm is a batch-based optimization algorithm, which calculates the gradients using the entire training dataset. The gradients are then used to update the weights of the network. The main advantage of this algorithm is that it is guaranteed to converge to a global minimum, but it can be very slow for large datasets.
    Stochastic Gradient Descent (SGD): Stochastic gradient descent is an optimization algorithm that uses random samples from the training dataset to estimate the gradients. Instead of using the entire dataset, it uses just a single example at each iteration to update the weights. This method is computationally more efficient than gradient descent, but it is also less accurate. It can be more sensitive to the choice of initial weights, and it may converge to a local minimum instead of a global minimum.
    Mini-batch Stochastic Gradient Descent (mini-batch SGD): Mini-batch stochastic gradient descent is a variant of stochastic gradient descent that uses a small, fixed-size subset of the training data, called a mini-batch, to calculate the gradients. It is a trade-off between the computational efficiency of SGD and the accuracy of gradient descent. It has been shown to converge faster and be more stable than pure SGD, and it is the most commonly used optimization algorithm for training neural networks.
51. [H] It’s a common practice to train deep learning models using epochs: we sample batches from data **without** replacement. Why would we use epochs instead of just sampling data **with** replacement?
    One reason is that the convergence rate of sampling without replacement is faster (https://arxiv.org/pdf/1202.4184v1.pdf, short explantion can be found here: https://stats.stackexchange.com/questions/235844/should-training-samples-randomly-drawn-for-mini-batch-training-neural-nets-be-dr).
    In addition, since we are training only one model (and not multiple like decision trees in a random forest), allowing the model to see as many examples as possible through sampling without replacement reduces bias and makes the model better at generalization.
52. [M] Your model’ weights fluctuate a lot during training. How does that affect your model’s performance? What to do about it?
    The fluctuation during training can be a sign the model struggles with convergence and that it has high variance. This can affect the model's accuracy and reliability. There can be a number of reasons why this happens:
        1. High learning rate: The weight updates take large steps in the direction of the gradient and creates fluctuation. Reducing the learning rate can help.
        2. Small batch size: The smaller the batch size the noisier the gradients which can cause fluctuation in the weight updates. Increasing the batch size or doing gradient accumulation can help. In gradient accumulation the weights are not updated after each batch, but after a number of preset batches are complete to reduce the noise.
53. Learning rate.
    34. [E] Draw a graph number of training epochs vs training error for when the learning rate is:
        1. too high
        2. too low
        3. acceptable.
    35. [E] What’s learning rate warmup? Why do we need it?
        Learning rate warmup is a technique used to gradually increase the learning rate during the initial stages of training. The idea is to start with a small learning rate and gradually increase it over a certain number of training steps or epochs.
        There are several reasons why learning rate warmup can be useful:
        High learning rate instability: When starting with a high learning rate, the model's weights can fluctuate a lot, leading to instability and poor performance. Learning rate warmup allows the model to converge to a stable solution before increasing the learning rate.
        Avoiding poor local minima: Starting with a high learning rate can cause the model to converge to a poor local minimum, rather than a global minimum. Learning rate warmup allows the model to explore the parameter space before settling into a suboptimal solution.
        Gradient sparsity: When the gradients are sparse, it can be hard for the optimizer to make progress with a high learning rate. A warmup period allows the optimizer to converge to a good initial point before increasing the learning rate.
54. [E] Compare batch norm and layer norm.
    Batch norm transforms the output of each layer based on the mean and variance of all the samples in the batch. In other words, it computes the mean and variance of each feature across all batch samples and trasforms each batch's feature value based on the calculated statistics. This means the the batch size and sequence length affects batch normalization. Also, because the statistics depend on all the batch samples, using batch norm in parallel settings is difficult.
    On the other hand, layer norm is independent on the batch and calculates the mean and variance for each sample separately, and is therefore better suited for when the sequence lengths are different in a batch or when the training is done in parallel.
    layernorm is more suitable for NLP tasks where the sequence lenghts vary. batch norm is more common in computer vision tasks.
55. [M] Why is squared L2 norm sometimes preferred to L2 norm for regularizing neural networks?
    Squared L2 norm has a smooth gradient everywhere, as opposed to L2 norm which has a kink at the origin. This helps with stable updates and faster convergence.
56. [E] Some models use weight decay: after each gradient update, the weights are multiplied by a factor slightly less than 1. What is this useful for?
57. It’s a common practice for the learning rate to be reduced throughout the training.
    36. [E] What’s the motivation?
    37. [M] What might be the exceptions?
58. Batch size.
    38. [E] What happens to your model training when you decrease the batch size to 1?
    39. [E] What happens when you use the entire training data in a batch?
    40. [M] How should we adjust the learning rate as we increase or decrease the batch size?
59. [M] Why is Adagrad sometimes favored in problems with sparse gradients?
60. Adam vs. SGD.
    41. [M] What can you say about the ability to converge and generalize of Adam vs. SGD?
    42. [M] What else can you say about the difference between these two optimizers? 