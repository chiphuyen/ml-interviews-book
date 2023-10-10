### 7.1 Basics

1. [E] Explain supervised, unsupervised, weakly supervised, semi-supervised, and active learning.
    Supervised learning is when the training data is labelled and the model can correct its learning using them. 
    Unsupervised learning is when there are no labels and the model has to learn from the inputs itself how to make predictions. Note that unsupervised learning in some settings are similar to supervised, e.g. masked language models.
    Weakly supervised learning uses heuristic functions to label data for training and does not require any labels from the training data.
    Semi-supervised learning requires some labelled data to train an initial model on. The model predicts labels on the unlabelled data and those with high raw probability get added to the training data. The process continues until the desired performance is reached.
    Active learning selects subset of data points to learn from. An active learner makes predicitons on unlabelled data and sends the ones it is least confident about to annotators to label.
2. Empirical risk minimization.
    1. [E] What’s the risk in empirical risk minimization?
        The goal of ERM is to minimize the risk of the model. The risk is defined as the expected value of the loss function over the true underlying data distribution. However, this risk is estimated by taking the average of the loss function over the training data. This creates the risk of the data distribution in the real world/production being different from the training data and therefore the model risk being higher than estimated. Another issue is that this estimate will be much lower if the model is overfitting. 
    2. [E] Why is it empirical?
        Because it is estimated from the data at hand, i.e. the training data as opposed to being computed on the true underlying data distribution.
    3. [E] How do we minimize that risk?
        Having more data to represent the true distribution and using regularization, early stopping, dropout to avoid overfitting.
3. [E] Occam's razor states that when the simple explanation and complex explanation both work equally well, the simple explanation is usually correct.  How do we apply this principle in ML?
    One way to apply Occam's razor in machine learning is by using simpler models, such as linear regression, instead of more complex models like deep neural networks. A simpler model will have fewer parameters to learn, and therefore less risk of overfitting the data.
    Another way to apply Occam's razor in machine learning is through feature selection. When dealing with large datasets, it is often the case that not all features are relevant to the problem at hand. By selecting only the most relevant features, you can reduce the complexity of the model and improve its performance.
    Additionally, techniques like regularization, such as L1, L2, and dropout, can also be used to reduce the complexity of the model and prevent overfitting.
4. [E] What are the conditions that allowed deep learning to gain popularity in the last decade?
    1. Cloud computing, more compute power: Availability of virtual machines with a wide variety of compute power, GPU resources and the easy access of all from cloud providers.
    2. Open-source tools: Access to powerful deep learning frameworks such as TF and Pytorch
    3. Open-source data: A lot more data is available which makes it easier for practitioners to get past the long data collection phase.
    4. Open-source models: Has enable collaboration and improvements upon released models.
5. [M] If we have a wide NN and a deep NN with the same number of parameters, which one is more expressive and why?
    A deeper NN. Because it can model more complex functions and has more opportunities to learn hierarchical representations of the data which makes it better at generalization. On the other hand, a wider network with more neurons is likely to memorize the inputs and the corresponding outputs rather than learn the underlying representation of the inputs.
6. [H] The Universal Approximation Theorem states that a neural network with 1 hidden layer can approximate any continuous function for inputs within a specific range. Then why can’t a simple neural network reach an arbitrarily small positive error?
    A neural network, like any other machine learning model, is only able to approximate the underlying function based on the data it has been trained on. This means that even if the network has the capacity to approximate the function, it can still make errors if the data it is trained on is noisy or incomplete. With only a single hidden layer, the model will not be able to learn the hierarchical representation of the data and will underfit. Another reason can be local minima, i.e. when the optimization algorithm gets stuck in a sub-optimal solution, preventing the network from reaching the global minimum.
7. [E] What are saddle points and local minima? Which are thought to cause more problems for training large NNs?
    Saddle points are critical points that are neither minima or maxima. This means that in some directions the objective function is increasing, while in other directions the objective function is decreasing.
    A local minimum is a point in the parameter space where the gradient of the objective function is zero. This means that the objective function has a minimum value in the neighborhood of that point and the gradient is zero in all directions. 
    At a saddle point, the gradient of the loss function is zero in some directions but not all. This means that the optimization algorithm is not decreasing the loss function in some directions, and it will not converge to a global minimum. Instead, the algorithm may oscillate or even diverge which makes the learning process unstable. So saddle points are generally more problematic than local minima.
8. Hyperparameters.
    4. [E] What are the differences between parameters and hyperparameters?
        Parameters are the internal variables of a model that are learned from the data during the training process, while hyperparameters are the external variables of a model that are set before the training process starts. Hyperparameter tuning is an essential step in the machine learning pipeline and it's necessary to find the optimal values for the hyperparameters that make the model perform well on unseen data.
    5. [E] Why is hyperparameter tuning important?
        Hyperparameter tuning is important because it allows you to optimize the performance of a model by adjusting the settings of the model that are not learned from the data. By systematically exploring the hyperparameter space, we can find the optimal values that make the model perform well on unseen data. This can improve the generalization of the model and prevent overfitting.
    6. [M] Explain algorithm for tuning hyperparameters.
        1. Grid Search: In grid search, all possible combinations of the hyperparameter values are trained and evaluated. This is a simple and straightforward method, but it can be computationally expensive for large hyperparameter spaces.
        2. Random Search: In random search, random combinations of the hyperparameter values are trained and evaluated. This is a more efficient method than grid search as it requires fewer evaluations, but it may not explore the hyperparameter space as thoroughly.
        3. Bayesian optimization: This algorithm is based on Bayesian statistics, it models the distribution of the hyperparameters and the objective function, and it uses this distribution to choose the next set of hyperparameters to evaluate. This algorithm is more computationally expensive than grid or random search, but it can converge faster to the optimal solution.
9. Classification vs. regression.
    7. [E] What makes a classification problem different from a regression problem?
        The main difference between classification and regression is the type of output they predict: classification is used to predict discrete labels or classes, while regression is used to predict continuous output values. 
    8. [E] Can a classification problem be turned into a regression problem and vice versa?
        Yes, a classification problem can be turned into a regression problem by converting the categorical output variable into a continuous one. For example, instead of predicting a class label, the model could predict a probability of the input belonging to each class. Similarly, a regression problem can be turned into a classification problem by converting the continuous output variable into a categorical one. For example, by dividing the range of output values into bins and assigning a class label to each bin, the model could predict the class label of the input instead of the continuous output value.
10. Parametric vs. non-parametric methods.
    9. [E] What’s the difference between parametric methods and non-parametric methods? Give an example of each method.
        Parametric methods are methods that learn using a pre-defined mapping function to map the inputs to the output. These methods make assumptions about the probability distribution of the data, typically assuming a normal distribution. In addition, these methods have a fixed number of parameters. An example of a parametric method is linear regression, which assumes that the relationship between the independent and dependent variables is linear, and estimates the coefficients of the line of best fit.
        Non-parametric methods, on the other hand, make fewer assumptions about the data distribution and do not have a fixed number of parameters or a pre-defined mapping function to learn the relationship between inputs and outputs. These methods are more flexible and can be applied to a wider range of data types. An example of a non-parametric method is the k-nearest neighbors (k-NN) algorithm, which classifies a data point based on the majority class of its k-nearest neighbors. Another example is decision trees.
    10. [H] When should we use one and when should we use the other?
        If we are not sure about the underlying data distributions, non-parametric is a better choice.
11. [M] Why does ensembling independently trained models generally improve performance?
    Becuase it combines the learning of different models with different biases. This combination generally makes the predictions more accurate. They also have the advantage of reducing overfitting because different models can make different predictions on the training data making the ensemble less susceptible to noise and better at generalization.
12. [M] Why does L1 regularization tend to lead to sparsity while L2 regularization pushes weights closer to 0?
    L1 regularization, also known as Lasso regularization, adds a penalty term to the loss function that is proportional to the absolute value of the weights. This penalty term encourages the weights to be small in magnitude, and it tends to lead to sparsity because it will drive some weights exactly to zero.
    L2 regularization, also known as Ridge regularization, adds a penalty term to the loss function that is proportional to the square of the weights. This penalty term encourages the weights to be small, but it does not encourage them to be exactly zero. Instead, it pushes the weights closer to zero, but not necessarily exactly to zero.
13. [E] Why does an ML model’s performance degrade in production?
    There can be a variety of reasons:
        1. Data drift
        2. Concept drift
        3. Different evironment settings from the dev environment, e.g. package versions
14. [M] What problems might we run into when deploying large machine learning models?
    1. Memory and computational resources: Large models require a lot of memory and computational resources to run, which can be a challenge for deployment on resource-constrained devices or in cloud environments with limited resources.
    2. Latency: Large models can have high latency, which can make them difficult to use in real-time or near real-time applications.
    3. Retraining and maintenance: Retraining and maintaining large models can be challenging, especially when deploying multiple versions of the same model or updating the model over time.
15. Your model performs really well on the test set but poorly in production.
    11. [M] What are your hypotheses about the causes?
        1. It can be that the training data is not representative of the production data, i.e. the user inputs are much noisier than the training data. 
        2. Alternatively, there can be data drift or concept drift. 
        3. The model may be biased towards certain groups or input values, resulting in poor performance for certain subpopulations.
        4. Preprocessing or feature engineering steps might vary between the environments
    12. [H] How do you validate whether your hypotheses are correct?
        Run invariance and slice-based tests on the training data and see if small pertubations on the trianing data affects the results or if the results vary depending on the sub-group. If it does it means that the model is not robust enough. Ideally these tests should have been run prior to making the model live.
        In the case of data drift we can compare the distribution of training and production data with statistical methods such as KL divergence.
    13. [M] Imagine your hypotheses about the causes are correct. What would you do to address them?
        You will need to retrain your model and include perturbed data in the training set to make it less susceptible to noise.
        In the case of data or concept drift you will need to retrain your model on new data or do online training to learn the new relationships.

### 7.2 Sampling and creating training data

1. [E] If you have 6 shirts and 4 pairs of pants, how many ways are there to choose 2 shirts and 1 pair of pants?
    There are 15 ways to choose 2 shirts out of 6, and 4 ways to choose 1 pair of pants out of 4. To find the number of ways to choose both items, you would multiply these two values: 15 x 4 = 60.
2. [M] What is the difference between sampling with vs. without replacement? Name an example of when you would use one rather than the other?
    Sampling with replacement means that after a sample is selected, it is put back into the population so that it can be selected again. Sampling without replacement means that once a sample is selected, it is not put back into the population and cannot be selected again. An example where one would pick sampling with replacement over without is boostrapping in bagging. On ther other hand for splitting the data into train and test sets you would want to sample without replacement.
3. [M] Explain Markov chain Monte Carlo sampling.
    The basic idea behind MCMC is to construct a sequence of samples, called a Markov chain, that is designed to converge to the target distribution. The chain starts at some initial state and then iteratively generates new states that are probabilistically determined by the current state. After running the chain for a sufficient number of iterations, the samples generated by the chain will be distributed according to the target distribution.
4. [M] If you need to sample from high-dimensional data, which sampling method would you choose?
    MCMC based methods are suitable for sampling from high dimensional data. e.g. Gibbs sampling, Metropolis-Hastings.
5. [H] Suppose we have a classification task with many classes. An example is when you have to predict the next word in a sentence -- the next word can be one of many, many possible words. If we have to calculate the probabilities for all classes, it’ll be prohibitively expensive. Instead, we can calculate the probabilities for a small set of candidate classes. This method is called candidate sampling. Name and explain some of the candidate sampling algorithms.
    1. Softmax sampling: instead of calculating the softmax for all possible classes, it samples them based on a distribution function Q and trains the model to maximize the probability of the target class over the sample set.
    2. Noise Contrastive Estimations: transforms the training to a binary logistic regression where the non-target classes are sampled and the goal is to predict whether the output belongs to the positive class, i.e. target or the noise.
    **Hint**: check out this great [article](https://www.tensorflow.org/extras/candidate_sampling.pdf) on candidate sampling by the TensorFlow team.
6. Suppose you want to build a model to classify whether a Reddit comment violates the website’s rule. You have 10 million unlabeled comments from 10K users over the last 24 months and you want to label 100K of them.
    1. [M] How would you sample 100K comments to label?
        1. Time-based sampling: this method is useful when the data is collected over a specific period of time and you want to ensure that the labeled dataset represents the whole period. You could sample comments from different months or years to ensure that the labeled dataset is diverse and covers the whole period of 24 months.
        2. User-based sampling (cluster sampling): this method is useful when you want to ensure that the labeled dataset represents the whole user base. You could randomly select 100 users and sample all their comments, or sample a random number of comments for each user. This way the labeled dataset represents a diverse group of users.
        3. High-uncertainty sampling: This method is useful when you want to ensure that the model is exposed to examples that are difficult to classify and have a low confidence level. You could sample comments that are difficult to classify as violating or non-violating the website's rule. For example, you could sample comments that are written in a different language, contain sarcasm or have a neutral sentiment. Alternatively, weakly supervised heuristics to label some samples enought to train a prelimenary model that we can use as an active learner to determine a set of samples that are more difficult to predict.
    1. [M] Suppose you get back 100K labeled comments from 20 annotators and you want to look at some labels to estimate the quality of the labels. How many labels would you look at? How would you sample them?
        A good starting point for the number of samples to inspect would be around 10% of the total number of labels. In this case, that would be around 10,000 labels. This sample size would be large enough to get a good estimate of the inter-annotator agreement while still being manageable to inspect.
        However, it's worth noting that this is just a starting point and the actual number of samples to inspect may need to be adjusted based on the results of the agreement metrics and the inspection of the sample of labels.
        For instance, if the inter-annotator agreement is found to be low, it might be necessary to inspect more labels to identify the sources of disagreement. On the other hand, if the agreement is high and the labels are found to be of high quality, it might be possible to inspect fewer labels to confirm the quality of the labels.
        Another alternative is random sampling, cluster sampling (group by user and then sample from each user), systemic sampling (pick every 10 comments)
    **Hint**: This [article](https://www.cloudresearch.com/resources/guides/sampling/pros-cons-of-different-sampling-methods/) on different sampling methods and their use cases might help.

7. [M] Suppose you work for a news site that historically has translated only 1% of all its articles. Your coworker argues that we should translate more articles into Chinese because translations help with the readership. On average, your translated articles have twice as many views as your non-translated articles. What might be wrong with this argument?
    **Hint**: think about selection bias.
    1. Selection bias: The sample of translated articles may not be representative of all articles on the site. The translated articles may be more popular or more likely to be viewed, which could be due to a number of factors, such as the topic, headline, or author.
    2. Causation vs correlation: The fact that translated articles have twice as many views as non-translated articles does not necessarily mean that translating more articles will lead to more views. There could be other factors that are causing the increased readership and translating more articles may not necessarily lead to more views.
8. [M] How to determine whether two sets of samples (e.g. train and test splits) come from the same distribution?
    There are different statistical tests for this:
        1. Maximum Mean Discrepancy (MMD): Is a kernal-based method that computes the distance between the means of two projections (distributions) in higher dimensions
        2. Kullback Leibler divergence (KL divergence) : KL divergence is a measure of how different two probability distributions are, it can be used to compare two sets of samples and determine if they come from the same distribution.
        3. Chi-Squared: It compares the observed frequencies of the two samples against the expected frequencies of the two samples if they come from the same distribution.
9. [H] How do you know you’ve collected enough samples to train your ML model?
    1. For more traditional models, a simple rule of thumb is to have 10 x data than the features.
    2. Investigating similar case studies on similar model architectures.
    3. Run cross validation on existing data, if overfitting occurs and the model is not that complex, can be indicating that more data is needed.
    4. Training an initial model and investigating its mistakes can be telling
10. [M] How to determine outliers in your data samples? What to do with them?
    1. Visual inspection: One way to detect outliers is to visually inspect the data by creating plots such as histograms, scatter plots, or box plots. Outliers will be represented as points that are far from the main cluster of points.
    2. Z-score: Z-score, also known as standard score, is a measure of how many standard deviations an observation is from the mean. A z-score greater than 3 or less than -3 can be considered as an outlier.
    3. Interquartile range (IQR): Interquartile range (IQR) is a measure of the spread of the data. It is defined as the difference between the 75th percentile and the 25th percentile. Data points that are more than 1.5*IQR below the 25th percentile or above the 75th percentile can be considered as outliers.
    4. Mahalanobis Distance: Mahalanobis Distance is a method that takes into account the correlation among the variables. It calculates the distance of each point from the mean of the data, taking into account the covariance matrix.

    What we do with outliers depends on the task at hand, depending on how big of an outlier the samples are and whether we think they can occur in production we can can delete them or keep them. Transformations like log can be performed to make the values more aligned with the rest of the values.
11. Sample duplication
    1. [M] When should you remove duplicate training samples? When shouldn’t you?
        Cases where they should not be removed:
            1. Duplicates should not be removed if the duplicated samples are representative of the real world, e.g. duplicated objects in a scene can be a common scenario and therefore they should be kept
            2. If the duplicates were created to oversample a minority class they should be kept to avoid bias towards the majority class
        In the case where duplicates are a result of a sampling error and do not represent the real world data distribution they should be removed as they can introduce unwanted bias towards certain data classes and increase the training time. 
        It's important to understand the undelying cause of duplicates before making a decision.
    1. [M] What happens if we accidentally duplicate every data point in your train set or in your test set?
        Duplicating the train set can have some negative effects. For example, it can lead to memorization of the duplicates and overfitting which will not generalize to unseen data. It can also increase the training time and memory requirements. 
        For the test case, if there is leakage, the model may have an inflated performance by "predicting" samples that were in the training set and memorized correctly. Duplicating the test data will most likely result in increased inference time overall but since the metrics are in ratios, predicting a sample wrong/correctly many times should not affect the overall metrics.
12. Missing data
    1. [H] In your dataset, two out of 20 variables have more than 30% missing values. What would you do?
        Deletion: the easiest but not so effective way is to delete those variables. It's probably better to delete the variables over the rows since the latter results in removing over 30% of the data.
        However, this results in less information to learn from. It is important to investigate why the values of each variable is missing. Are they completely at random? Not at random, or at random? If they are not missing at random, the fact that they are missing can be quite telling and keeping them will be better. In this case setting the missing values to default values that are different from acceptable values can be a better solution.
    1. [M] How might techniques that handle missing data make selection bias worse? How do you handle this bias?
        In the case where the data is missing at random, i.e. missing not because of the true missing value but because of some other value, we can introduce selection bias by deleting the samples. For example, if the participants from gender A do not disclose their age and we delete all the rows with missing age, we delete all the samples from gender A. This makes selection bias worse because now the model doesn't see examples of this population and will underperform on this sub-population in production. 

        It's best to handle the missing data differently to not make selection bias worst. In the case where there are still samples with the rarer feature values, e.g. gender A from the example above, might be worth oversampling them to have more examples from the sub-group.
13. [M] Why is randomization important when designing experiments (experimental design)?
    It is important in selecting a population that is representative of the true distribution. This way, we don't introduce unwanted bias from selecting a disproportionate number of smaples from a certain population. 
14. Class imbalance.
    1. [E] How would class imbalance affect your model?
        It can lead to the majority class dominating the predictions which has a negative affect on correctly classifying the samples from the minority classes.
    1. [E] Why is it hard for ML models to perform well on data with class imbalance?
        In an imbalanced dataset the model becomes biased towards the majority class as it mostly sees samples of that class and does not see enough of the minority classes. This doesn't give it enough information to learn the underlying patterns of the rare class and so it becomes less sesitive to it.  
    1. [M] Imagine you want to build a model to detect skin legions from images. In your training dataset, only 1% of your images shows signs of legions. After training, your model seems to make a lot more false negatives than false positives. What are some of the techniques you'd use to improve your model?
        1. Data augmentation: More samples from the rare class can be generated by slightly modifying the existing data, e.g. flipping, cropping, rotating.
        2. Depending on the number of majority samples you can downsample them or look into oversample methods for the lesion class.
        3. The cost function can be modified to account penalize the model more when it makes a FN. One way is to give each class a weight that is the inverse of the number of samples in that class. Another useful cost function used for class imbalance is Focal loss which gives a higher weight to hard examples, i.e. the minority examples by multipying the log p with (1 - p)^gamma.
15. Training data leakage.
    1. [M] Imagine you're working with a binary task where the positive class accounts for only 1% of your data. You decide to oversample the rare class then split your data into train and test splits. Your model performs well on the test split but poorly in production. What might have happened?
        Oversampling can be duplicating the samples or tweaking some features by a very little cmount. The oversampling should have happened after splitting the data. With doing it prior to splitting, the data from train (oversampled points) has leaked to the test set and the model is overly optimistic about its performance on test. The production data is different from the training and testing data which explains the discrepency. 
    1. [M] You want to build a model to classify whether a comment is spam or not spam. You have a dataset of a million comments over the period of 7 days. You decide to randomly split all your data into the train and test splits. Your co-worker points out that this can lead to data leakage. How?
        Splitting the data randomly into train and test splits can lead to data leakage when working with a time-series dataset, such as comments over a period of 7 days, because the model may learn patterns related to specific trends of the day. This can lead to information from the future leaking into the training and allowing the model to cheat during evalutation.
            
    **Hint**: You might want to clarify what oversampling here means. Oversampling can be as simple as dupplicating samples from the rare class.

16. [M] How does data sparsity affect your models?
    1. Increased space and time complexity
    2. Bias towards dense features and underestimating the predictive power of sparse features
    3. Overfitting
    
    **Hint**: Sparse data is different from missing data.

17. Feature leakage
    26. [E] What are some causes of feature leakage?
        Using the target to construct features is one way where the features have information about the target that should not be accessible. Another cause can be that the data is not representative of the real world examples and the model learns features from the data that are highly correlated with the target but they do not generalize well, e.g. the neural network to classify huskies and wolves only saw examples of wolves with a snowy background and huskies with a green background. 
    27. [E] Why does normalization help prevent feature leakage?
        It can help reduce the correlation between leaked features and the target.
    28. [M] How do you detect feature leakage?
        1. Investigating the correlation between the target and features or combination of features.
        2. Ablations studies to find features that significately affect performance and identifying the cause.
        3. Keeping an eye on new features if they significantly improve the model performance.
18. [M] Suppose you want to build a model to classify whether a tweet spreads misinformation. You have 100K labeled tweets over the last 24 months. You decide to randomly shuffle on your data and pick 80% to be the train split, 10% to be the valid split, and 10% to be the test split. What might be the problem with this way of partitioning?
    With a random shuffle rather than splitting based on dates there will be temporal data leakage from test splits to train and validation and the model will essentially cheat in predicting the test data, but perform poorly on production data.
19. [M] You’re building a neural network and you want to use both numerical and textual features. How would you process those different features?
    The textual features need to be tokenized into smaller units, e.g. words, and preprocessed, they then will be mapped to different numbers and passed to an either trainable embedding layer or a pretrained layer, e.g. GloVe. After the textual data is embedded, we can concatenate the normalized numerical data to it.
20. [H] Your model has been performing fairly well using just a subset of features available in your data. Your boss decided that you should use all the features available instead. What might happen to the training error? What might happen to the test error?
    1. Training error: The training error may decrease as the model has more information to learn from. With more features, the model may be able to better fit the training data, leading to a lower training error.
    2. Test error: The test error may increase or decrease depending on whether the additional features are relevant or not. If the additional features are relevant to the task and provide useful information, the test error may decrease. However, if the additional features are not relevant or contain noise, the test error may increase. However, if the features are not informative, it can lead to overfitting and the test error will increase.

    **Hint**: Think about the curse of dimensionality: as we use more dimensions to describe our data, the more sparse space becomes, and the further are data points from each other.

### 7.3 Objective functions, metrics, and evaluation

1. Convergence.
    1. [E] When we say an algorithm converges, what does convergence mean?
        In the context of algorithms, convergence refers to the process of approaching a specific value or set of values as the number of iterations increases. For example, in optimization problems, an algorithm is said to converge when the solution it finds is within a specified tolerance of the true optimal solution, or when the difference between solutions in consecutive iterations is below a certain threshold. In machine learning, an algorithm is said to converge when the performance of the model on the training set stops improving or even starts degrading with more training examples. The specific definition of convergence can vary depending on the context and the algorithm in question.
    1. [E] How do we know when a model has converged?
        When the loss does not change much from one iteration to the next.
1. [E] Draw the loss curves for overfitting and underfitting.
    https://docs.aws.amazon.com/machine-learning/latest/dg/model-fit-underfitting-vs-overfitting.html
1. Bias-variance trade-off
    1. [E] What’s the bias-variance trade-off?
        The bias-variance trade-off is a fundamental concept in machine learning that refers to the trade-off between a model's ability to fit the training data well (low bias) and its ability to generalize well to new, unseen data (low variance). A model with high bias is one that makes strong assumptions about the underlying relationship between the input and output variables, and as a result, may not fit the training data very well. On the other hand, a model with high variance is one that is highly sensitive to the specific details of the training data, and may not generalize well to new data. In general, a good machine learning model will strike a balance between these two extremes, and the goal of many machine learning techniques is to find the right balance between bias and variance.
    1. [M] How’s this tradeoff related to overfitting and underfitting?
        A model with high bias is said to be underfitting, because it is not able to fit the training data well. A model with high variance is said to be overfitting, because it is fitting the noise in the training data, rather than the underlying pattern
    1. [M] How do you know that your model is high variance, low bias? What would you do in this case?
        If a model has high variance and low bias, it means that it is highly sensitive to the specific training data it was trained on. It's likely to perform well on the training data but poorly on unseen data, overfitting the training data. This can be identified by observing that the model has high performance on the training set but poor performance on the validation or test sets.
        To address this issue, one can use techniques such as regularization, which adds a penalty term to the loss function to discourage large weights, or ensemble methods, which combine the predictions of multiple models to reduce variance. Another strategy could be to collect more data to increase the size of the training set.
    1. [M] How do you know that your model is low variance, high bias? What would you do in this case?
        A model that is low variance and high bias generally means that the model is underfitting the data. This can happen if the model is too simple, or if the model has not been trained for enough iterations. To address this issue, one could try to increase the complexity of the model
1. Cross-validation.
    1. [E] Explain different methods for cross-validation.
        1. K-fold cross-validation: The data is divided into k subsets, and the model is trained and evaluated k times, each time using a different subset as the evaluation set and the remaining subsets as the training set.
        2. Leave-one-out cross-validation: This method is similar to k-fold cross-validation, but with k set to the number of samples in the data. For each iteration, one sample is used as the evaluation set, and the remaining samples are used as the training set.
        3. Stratified cross-validation: This method is used when the data is imbalanced, meaning there are unequal number of samples in each class. The data is divided into k subsets, ensuring that each subset has roughly the same class distribution as the original data.
    1. [M] Why don’t we see more cross-validation in deep learning? 
        The main reason why we don't see more cross-validation in deep learning is that deep learning models are computationally expensive to train, and performing cross-validation would require training multiple versions of the same model, which can be computationally infeasible. Additionally, in deep learning, we often use large amounts of data to train models, making cross-validation less necessary. A single split of the data into training and test sets is often sufficient for evaluating the performance of a deep learning model.
1. Train, valid, test splits.
    1. [E] What’s wrong with training and testing a model on the same data?
        Because it will overfit to the data and not generalize. The goal of testing the trained model is to see how it can generalize on unseen data and make sure that the learned parameters do well on data it has not used to adjust those parameters. Using the same data defeats this purpose.
    1. [E] Why do we need a validation set on top of a train set and a test set?
        A validation set is used to tune the hyperparameters of a model during the training process. It allows us to evaluate the performance of the model on unseen data before it is tested on the test set. Without a validation set, we would only have the training set to tune the hyperparameters, and there's a risk of overfitting. 
    1. [M] Your model’s loss curves on the train, valid, and test sets look like this. What might have been the cause of this? What would you do?
        It seems that the validation loss is increasing which is a sign of overfitting. Interestingly the overfitted model generalizes well to the test data which can be from data leakage. Another explanation can be that the validation dataset has a different distribution from the train and test. Some solutions would be to use regulaization and early stopping to avoid overfitting. In addition, it will be useful to investigate the distribution of the data in the different sets to see if validation is different from train and test. If so, the splits should be different for the train and test to be representative of the differences. 
        Investigation on data leakage is also necessary and the leaked examples should be removed from the train set.

  <center>
    <img src="images/image25.png" width="60%" alt="Problematic loss curves" title="image_tooltip">
  </center>

1. [E] Your team is building a system to aid doctors in predicting whether a patient has cancer or not from their X-ray scan. Your colleague announces that the problem is solved now that they’ve built a system that can predict with 99.99% accuracy. How would you respond to that claim? 
    I would ask what the accuracy is in each class. It is very likely that the model is predicting no cancer most of the time and since the negative class is more prevalent, the accuracy is high. But we care more about detecting the cancer cases which are rarer.
1. F1 score.
    1. [E] What’s the benefit of F1 over the accuracy?
        In situations where the data is imbalanced, F1 score is a better measure compared to accuracy because it gives equal weight to the precision and recall. In such cases, accuracy can be misleading as the classifier might just predict the majority class and get a high accuracy even though it's not doing a good job at classifying the minority class.
    1. [M] Can we still use F1 for a problem with more than two classes. How?
        Macro and Micro F1-scores are used for multi-class probelems. Macro-F1 is the average of F1 scores of each class. Micro-F1 is the similar to F1 but sums the TP, FP and FNs across all classes which is the same as accuracy. Micro-F1 gives every observation equal weight and is not suitable for imbalanced classes, whereas macro F1 gives each class equal weight and is better for imbalanced classes.
        Good resources: https://stackoverflow.com/questions/37358496/is-f1-micro-the-same-as-accuracy, https://stephenallwright.com/micro-vs-macro-f1-score/
1. Given a binary classifier that outputs the following confusion matrix.
  <table>
    <tr>
     <td>
     </td>
     <td>
  Predicted True
     </td>
     <td>Predicted False
     </td>
    </tr>
    <tr>
     <td>Actual True
     </td>
     <td>30
     </td>
     <td>20
     </td>
    </tr>
    <tr>
     <td>Actual False
     </td>
     <td>5
     </td>
     <td>40
     </td>
    </tr>
  </table>

  1. [E] Calculate the model’s precision, recall, and F1.
    Precision (PPV) = 30 / (30 + 5) = 0.857
    Recall (Sensitivity, TPR) = 30 / (30 + 20) = 0.6
    F1 = 2 * (Precision * Recall) / (Precision + Recall) = 2 * (0.857 * 0.6) / (0.857 + 0.6) = 0.69
  1. [M] What can we do to improve the model’s performance?
    1. Modifying the decision threshold will affect the number of FP and FN, e.g. reducing it will decrease the number of FN but likely increase the number of FPs. Depending on the data the overall F1 score can increase. 
    2. Investigating the feature values of FN and FP cases and seeing if any modifications to the features can help
    3. Using other classifiers
    4. Changing the hyper-parameters
1. Consider a classification where 99% of data belongs to class A and 1% of data belongs to class B.
    1. [M] If your model predicts A 100% of the time, what would the F1 score be? **Hint**: The F1 score when A is mapped to 0 and B to 1 is different from the F1 score when A is mapped to 1 and B to 0.
        If A is mapped to 1, the precision will be 99% and the recall will be 100% so the F1 is 99.5%. However if A is mapped to 0, the F1 score will be undefined because the precision has a devision by 0. 
    1. [M] If we have a model that predicts A and B at a random (uniformly), what would the expected F1 be?
        In this case the predictions are 50-50, when A is class 1, the precision is 1 and the recall is 50 / 99 = .51 which gives an F1 score of 68%. In the opposite case, the precision is 1 / 50 and the recall is 1 which gives an F1 score of 3.9 %.
1. [M] For logistic regression, why is log loss recommended over MSE (mean squared error)?
        Because it is a probabilistic loss function that measures the dissimilarity between the predicted probability distribution and the true distribution and is a natural choice for logistic regression because the output of a logistic regression model is a probability, and log loss can directly measure the dissimilarity between the predicted probabilities and the true labels.
        Also, MSE is non-convex for logistic regression which makes finding the best fit to the data harder than log loss which is convex.
1. [M] When should we use RMSE (Root Mean Squared Error) over MAE (Mean Absolute Error) and vice versa?
        RMSE penalizes higher differences (larger errors) more. It is also differntiable everywhere. MAE on the other hand is more interpretable.
        RMSE is more suitable for cases where making large errors can be catashtrophic, e.g. fraud detection but in the case where the data has outliers for the results to not be  dominated by those data points MAE is a better choice.
1. [M] Show that the negative log-likelihood and cross-entropy are the same for binary classification tasks.
    Let's consider a binary classification task with two classes: class 0 and class 1, and let y be the true label (0 or 1) and p be the predicted probability of the positive class (class 1).

    The negative log-likelihood loss function is defined as:

        L(y, p) = -(y * log(p) + (1-y) * log(1-p))

    The cross-entropy loss function is defined as:

        H(y, p) = SUM(-y * log(p))
    which is the same as the negative log-likelihood.
1. [M] For classification tasks with more than two labels (e.g. MNIST with 10 labels), why is cross-entropy a better loss function than MSE?
    The MSE loss function compares the predicted output with the true output element-wise and calculates the mean squared error between them. The problem with this is that it assigns equal penalties to all the incorrect class predictions regardless of how confident the model is in those predictions. This can lead to the model being more cautious and not putting enough weight on the correct class, which is not ideal for multi-class classification problems.
    On the other hand, the cross-entropy loss function uses the predicted class probabilities to penalize the model for incorrect class predictions. It calculates the negative log-likelihood of the true class given the predicted class probabilities. The closer the predicted class probabilities are to the true class, the lower the cross-entropy loss will be. This means that the model will be penalized more for predictions that it is very confident in but are incorrect, and less for predictions that it is uncertain about. This encourages the model to be more confident in its predictions, which is desirable in multi-class classification.
1. [E] Consider a language with an alphabet of 27 characters. What would be the maximal entropy of this language? 
    log(27)
1. [E] A lot of machine learning models aim to approximate probability distributions. Let’s say P is the distribution of the data and Q is the distribution learned by our model. How do measure how close Q is to P?
    Kullback-Leibler divergence (KL divergence): a measure of the difference between two probability distributions, often used when the true distribution P is unknown. A symmetric version of this tests is the JS divergencet. Another measure for discrete values is Chi-squared distance.
1. MPE (Most Probable Explanation) vs. MAP (Maximum A Posteriori)
    1. [E] How do MPE and MAP differ?
        Considering the prior information (evidence), MAP finds a subset of non-evidence parameters with the highest probability. MPE finds the values for all non-evidence parameters with the highest probability.
    1. [H] Give an example of when they would produce different results.
        https://www.quora.com/What-are-the-cases-in-which-Most-Probable-Explanation-MPE-tasks-do-not-generalize-to-Maximum-A-Posteriori-MAP-task
1. [E] Suppose you want to build a model to predict the price of a stock in the next 8 hours and that the predicted price should never be off more than 10% from the actual price. Which metric would you use?
    Mean absolute percentage error is a metric that computes the average of the ratio between the absolute error and the actual value which is what we need in this case. 

    **Hint**: check out MAPE.

