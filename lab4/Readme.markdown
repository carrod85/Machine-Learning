# Lab4
## Description of business problem
First of all, It is important to be aware of the task to solve. The goal of this lab is to work with clustering models, 
which are by definition unsupervised methods that attempt to detect patterns in unlabeled data.


## Describe your approach
My intention has been to find first an easy to understand dataset that allowed me get a better understanding
the techniques of clustering without spending a lot of time understanding the data.

Looking for different examples online, I was unsure about how to make the selection of features
for clustering, and realizing that it was not very convenient to do it manually, I looked for documentation about the topic. 

I found out that a good technique for this feature selection was to use silhoutte score and therefore
being able to select the more relevant features that better fit for clustering without creating excesive noise.

looking for examples I run into a very well explained example that I used as a base for the task.
(link to description) https://programminghistorian.org/en/lessons/clustering-with-scikit-learn-in-python

The process followed has been the following:  
1. Download data set, studying it using the descriptive prints.
2. Taking the suggestion of author to remove data > quantile 0.9 since it is very far away of mean.
3. Scaling the data, so  one particular feature doesn't impact more than other because of that.
4. Using function progressiveFeatureSelection we can try different configuration with a good silhoutte.(This is descripted on the
following link https://perma.cc/K5PD-GQPQ)
5. Elbow method to check number of natural clusters.
6. Cluster plotting and PCA visualization of clusters for selected features.


## Describe your results selection
 The clear evidence that I have achieved a solution is that when plotting the clusters
 we can see that they are quite well segmented without barely noise.

## Evaluate and describe your results.
 I think a satisfactory solution has been found. Despite
 of using PCA and losing our initial features and creating new ones 
 that are somewhat nebulous to us, as they do not allow us to look at 
 specific aspects of our data anymore (such as word counts or known works).
 We are using the built-in function pairplot from seaborn module that allow us
 to valorate the clusters in regards to the selection of features. With the help of this 
 function we can certainly measure the weight of every feature and their relation with its cluster and
 the other features. Studying the dataset we see that it is quite decent.





