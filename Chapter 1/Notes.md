# Preface
## How the book is organized
### Part I: Fundamentals of Machine Learning
- What machine learning is?
- The steps in a typical machine learning project
- Basics of preparing data
- All the machine learning algorithms for different types of problems

### Part II: Neural Networks and Deep Learning
- What are neural nets?
- Building and training neural nets using Tensorflow and Keras
- Techniques and types of neural nets

-----
# Chapter 1: Machine Learning Landscape
Before we look through the specific locations on the map, let's get an overview of all the topics first:
1. Supervised Learning vs. Unsupervised Learning (and their variants)
2. Online vs. Batch Learning
3. Instance-based vs. model-based learning

We will also look at the typical workflow of each machine learning project. 

## What is Machine Learning?
The science of programming computers, so that they can learn from data. When the model is being trained, the example dataset that it has is called a training set. Each training example is called a training instance. The part of a machine learning system that learns and makes predictions is called a model.

## Why Use ML?
Take the example of a spam detection system. If coding it conventionally, we would have to find the typical patterns in spam on our own, and then hard code each case in to generate a proper filter. Since the problem is difficult, your program will likely be a long list of complex rules - pretty hard to maintain. In contrast, a machine learning solution learns and updates on its own, leading to a more maintainable and efficient solution, and most times more accurate.

In summary, ML is great for:
- Problems where existing solutions require a lot of fine-tuning or long lists of rules
- Complex problems for which using a traditional approach yields no good solution
- Fluctuating environments
- Getting insights into complex problems and large amounts of data

## Types of Machine Learning Systems
Here are the three general categories we divide the systems in:
1. How is the system supervised during training? (supervised, unsupervised, semi-supervised, self-supervised, etc.)
2. Whether the system can learn incrementally on the fly (online vs. batch learning)
3. Whether they work by simply comparing new data points to known data points, or instead by detecting patterns in the training data and building a predictive model (instance-based vs. model-based)

The categories can be combined in any way.

### Training Supervision
**Supervised Learning**: The dataset we feed contains *labels*, meaning that for each training instance the target is given. If we are predicting car prices, then in our training set we have all the fields plus the car price for the model.
**Unsupervised Learning**: Training data is *unlabeled*, no target variables given. A typical system for this type would be clustering, which finds patterns and separates the data into groups. We also have other problems like anomaly detection, where we can do things like detecting credit card fraud by checking for anomalies in the data.
**Semi-supervised Learning**: Since labelling data is time-consuming and costly, datasets would often have many missing labels. Some algorithms can deal with data that is partially labeled. Not too common, but an example is Google Photos. At first, it classifies images into different people on its own (unsupervised), then the user can input a label saying "Hey this is person B", and now the algorithm has a label to work with (supervised).
**Self-supervised Learning**: Another approach is to generate a labeled dataset from a fully unlabeled one. An example is a large set of unlabeled images of pets. You can randomly mask a part of the image, and then train a model to recover the original image. Hence, in training, the masked images are used as inputs and the original images are the labels.
**Reinforcement Learning**: Very different. The learning system (agent) can observe the environment, select and perform actions, and get rewards or penalties in return. It must then learn by itself what is the best strategy (policy) to get the most reward over time. A policy defines what action the agent takes in a given situation.

### Batch Vs. Online Learning
This criterion is to check whether a model can learn on the fly with a stream of incoming data.

**Batch Learning**:
System is incapable of learning on the fly. The system is trained on the full dataset *offline*, and then launched into production where it stops learning and operates based on the previous training only. This often causes *model rot*, which is because the rest of the world evolves while the model does not. We could continue training it offline every week or so and then pushing it into production, but that is inefficient. If the model relies on volatile data, it is better to go with online systems.

**Online Learning**:
Train the system incrementally by feeding it data instances sequentially, either individually or in small groups called mini-batches. Each learning step is fast and cheap, so the system can learn on the fly. These systems are great for problems that rely on adapting to changing data. It also reduces the load on the computer during training, instead of all of it being trained once, we let it sequentially go one by one on different batches until it's done all the sets.

The rate at which the system adapts to the changing data is called the *learning rate*. A higher learning rate would mean the system changes rapidly. A big problem with online learning is the tendency to train on bad data. If a bug comes up, or bad data, the model will train on it and will decline on its performance and other metrics. Users will often see the difference. Thus, we may want to watch out for bad inputs using techniques like anomaly detection.

### Instance-Based vs. Model-Based Learning
This category shows how the model generalizes the output.

**Instance-Based Learning**:
The system learns the examples (training data) by heart, and then generalizes the new cases by using a similarity measure to compare them to the learned examples.

**Model-Based Learning**:
Another way is to build a model from the examples and generalize from there. The book doesn't really give a clear definition of this, but how I understand it is that we find a relation between all the parameters from the training set. Like target = x1 + x2. Then, we apply that relationship onto the new cases to get the generalized outputs, instead of just comparing the new case to the old ones. 

## Main challenges of Machine Learning
The two major problems are either "bad model" or "bad data". Here are some bad data problems:

### Insufficient Quantity of Training Data
Even for simple problems, we often need thousands of training instances. For more complex ones, it can stretch to millions.

### Non-representative Training Data
Basically, the training data should try and include instances that correlate to the new instances we generalize for. If our dataset stops at a GDP per capita of 35,000, and then one of our generalizations is above that, then our data is not perfectly representative. It is very crucial to have representative data, and problems often arise with small sample sizes (contain sampling noise), but even very large datasets can be non-representative if the sampling method is flawed (called *sampling bias*).

### Poor-Quality Data
Obviously, if the training data has errors, outliers and noise, it is going to make it harder for the model to find the patterns, costing the overall performance. To fix this, most of the time is actually spent on cleaning the dataset before actually using it.

### Irrelevant Features
The system can only learn if the relevant fields are present. Hence, it is critical we use only the most important features to train on. This process is called *feature engineering*, and goes like this:
1. Feature Selection: select the most useful ones to train on
2. Feature Extraction: combine existing features to produce more useful ones
3. Creating new features from the existing ones

Now that we have looked at all the bad data problems, let's take a look at the bad model parts:

### Overfitting the Training Data
Essentially, this is when the model performs incredibly well on the training data, but not on generalizing the new cases. Often times, it misses the actual underlying pattern of the data, and instead focuses on the individual plot points within a training set. Take the example from the book. You may have been stolen from by taxi drivers many times, but that does not mean that every taxi driver is a thief. Our model can learn to over-generalize like that.

Technically speaking, overfitting happens when the model is too complex relative to the amount of noisiness in the training data. Here are some of the solutions:
1. Simplify the model by using fewer parameters, or by reducing the number of attributes the model uses to train
2. Gather more training data (leading to less chance of noise)
3. Reduce the noise in the training data (fix data errors and remove outliers)

Constraining a model to make it simpler and reduce the risk of overfitting is called *regularization*. It is controlled through *hyperparameters*, which are parameters that affect the learning algorithm, not the model being trained.

### Underfitting the Training Data
Opposite of overfitting; model is too simple, and the model does not get the underlying structure of the data. So we need to optimize it better for the problem at hand

----
## Testing and Validating
How do we know if a model does well? Often, we split the training dataset to two parts: the training set and the test set. You train on the train set, then test with the test set. Then you check for the generalization error after testing with the test set by comparing the model outputs to the actual labels. 

### Hyperparameter Tuning and Model Selection
It is hard often to find the best model for a problem. We just have to try them all out before going with one. But the work is not done once we find a model. We now need to tune it to prevent overfitting by regularizing, meaning adjusting the hyperparameters. Often, we simply run the algorithm over and over again, adjusting the hyperparameter and validating with the test set until we get a low error. Problem with this is that, now we have overfitted the learning system to the test data by checking for the error on it. So, this algorithm in production will not actually perform well. So we have to find an alternative approach.

A common solution is *holdout validation*. Simply, split the dataset into another part: the validation set. So, you tune the hyperparameters to perfect on the test set, then test and adjust again for the validation set.

However, splitting the dataset might cause us to either lose out training power or validating power with either being smaller compared to the other in certain cases. To avoid this, we often use techniques like cross-validation, which means using many small validation sets.
