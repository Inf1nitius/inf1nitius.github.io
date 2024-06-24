# Introduction
## Concepts
- Databases and Big Data
  - Relational data
  - Tables
  - SQL
  - Non-structured (Videos, Messages etc.)
- Statistics
  - Probability
  - Hypothesis testing
  - Neural network
- Machine Learning
  - Algorithms that automatically analyse large data sets
- Computing
- Visualisations
  - Line, area chart for time series
  - Bar chart for counts of different categories
  - Pie chart for proportions
## Resources
- Andrew Ng, Stanford CS 229 lecture notes, http://cs229.stanford.edu/notes/
- Ian Goodfellow et al. (2016), Deep Learning, Massachusetts Institute of Technology, http://deeplearningbook.org
- [BigML](https://bigml.com)

# Fundamental Notions in Data Science
## Spaces and Dimensions
### Curse of Dimensionality
- Hughes effect
- Exponential increase in volume when more dimensions added
- Makes high dimensionality data difficult to work with since adding more features increases noise
- Exponentially harder to find a pattern given amount of data (samples) remains the same
- Causes increase in complexity/running time in training; overfitting; number of samples required
### Overfitting
- Unable to generalise to new data
### Number of Samples
- When number of features (dimensions) increases, data becomes more sparse
- If $n$ samples are dense enough in $1D$, $n^d$ samples required in $dD$
- Thus method with more features likely to perform worse
- With "better" features, less dimensions required
### Dimensionality Reduction
- Results in increase in efficiency, classification performance, ease of interpretation and modelling; decrease in measurement, storage, computation costs

# Artificial Intelligence and Machine Learning
## Terminologies
### Artificial Intelligence (AI)
Cutting-edge machine capability
### Machine Learning (ML)
Recognising patterns without needing explicit instructions
### Representation Learning (RL)
Branch of ML where models learn from data automatically
### Artifical Neural Network (ANN)
Collection of artifical neurons inspired by biological brain cells
### Deep Learning (DL)
ANN consisting of multiple (at least 5) layers of artifical neurons
- 1 input
- 3 hidden
- 1 output
### Machine Vision (MV)
AI which recognises objects by appearance
### Natural Language Processing (NLP)
AI which interacts with natural language
## Unsupervised
- Data/input $x$ given without outcome $y$
- Goal is for model to discover a hidden, underlying structure within the data
- Works with unlabelled data
- Tasks are clustering (finding groupings using similarities or differences) and dimensionality reduction
## Supervised
- Data/input $x$ and outcome (to be predicted) $y$ given
- Goal is for model to learn a function that uses $x$ to approximate $y$
- Needs labelled data with "correct" output
- Tasks are either classification (discrete) or regression (continuous)
## DL
- Inspired by structure and function of the brain
- Neurons process information and integrate the influences across other neurons
- Biological neuron is made up of certain components:
  - Dendrites receive information from many other neurons
  - Cell Body aggregates this information via changes in cell voltage
  - Axon transmits a signal if voltage crosses a threshold
## Generative Adversarial Networks (GAN)
- Type of DL where 2 ANNs work against each other as adversaries
- *Generator* produces data (e.g. images)
- *Discriminator* distinguishes fake (generated) data from real
## RL
- No initial data input into algorithm
- An agent takes actions and receives direct feedback (rewarded/penalised) from the environment

# Unsupervised Learning: Clustering
## Concepts
### Clustering
- Groups data points into clusters such that:
  - Points within same cluster are similar
  - Points across clusters are different
- Data points grouped based on a similarity measure (e.g. Euclidean distance)
- Uses:
  - Market research by partitioning consumers into market segments
  - Social network analysis by recognising communities within large groups of people
  - Healthcare by grouping people based on health conditions to determine healthcare solutions
  - Recommender/Recommendation systems to segments consumers and products
## K-Means
### How it works
1. Define $k$ number of clusters
2. Assign arbitrary $k$ starting centroids
3. Allocate points to each centroid by proximity (e.g. Euclidean distance)
4. Calculate existing centroids of clusters as new "starting" centroids
5. Repeat 3-4 until boundary of clusters do not change (or threshold met)
### Challenges
- Attempts to minimise SSE (Euclidean distance) which is computationally difficult
- Not guaranteed to converge to optimal solution since different starting seeds (may) produce different results
  - Can be done multiple times with different starting seeds and choose model with smallest SSE
### Number of Clusters
- Elbow method
- G-Means clustering
  - Automatically discover $k$
  - Repeatedly test whether data in neighbourhood of centroid looks Gaussian (normal) and splits data if not
### Which Variables
Variables should be relevant to reason for clustering and should not have missing values across many observations
## Types of Clustering Algorithms
- K-Means
- Hierarchical Cluster Analysis (HCA)
  - G-Means (Gaussian)
- Gaussian Mixtures (EM-algorithm)

# Supervised Learning: Regression with Linear Models

## Linear Regression
- Linear approach for modelling the relationship between independent variable(s) $x$ and dependent variable $y$
- Assumes linear independence
- 2 types
  - Simple: 1 $x$ and 1 $y$
  - Multivariate: >1 $x$ and 1 $y$
- Provides mathematical way of finding impact of $x$ on $y$
- Allows analysis of statistical significance of each relationship
## Correlation
A measure of degree of linear associativity between 2 variables using Pearson's Correlation Coefficient
## Simple Linear Regression
- Forms a relationship $y=mx+c$
- Uses Method of Least Squares which minimises SSE between data and fitted line
- May not be able to capture or explain certain parts
### Assumptions
- Linearity: $x$ and $y$ have a linear relationship (proportional) and can be graphically represented with a straight line
- Errors: Errors are normally distributed, have equal variance, are independent
- Outliers: Point(s) which lie far away from computed regression line which affects calculation of least squares
### Appropriateness
Visualisations used first to understand the data (whether it should fit a straight line)
### Accessing the Regression Line
Coefficient of Determination $R^2$
- Ranges from 0 - 1 where higher $R^2$ shows high linear associativity and vice versa
## Multivariate Linear Regression
Involves more than one $x$ to predict $y$
- $y=m_1x_1+...+m_nx_n+c$
### Multicollinearity
- Occurs when there is high corelation between some of the independent variables
- Leads to less reliable results
- Number of independent variables should be reduced
- Curse of Dimensionality
### Accessing the Regression Line
- $R^2$ in general shows the quality of the fit to the data but can be increases simply by adding more features
- Adjusted $R^2$ is used instead as it adjusts for number of variables

# Supervised Learning: Classification with Linear Models

## Logistic Regression
- Used when linear regression is not appropriate (based on visualisation)
## Sigmoid (Logit) Function
$\sigma(z)=\frac{e^z}{e^z+1}=\frac{1}{1+e^{-z}}$

Used when concerned with dual outcomes
## Logistic Regression Function
Models the log of the outcome = ln of the odds of the outcome = $log(\frac{P}{1-P})$
## Curve fitting using Maximum Likelihood Estimator (MLE)
"Finds"and "reports" as statistics the parameters that are most likely to have produced the data
## Evaluation Metrics
### Confusion Matrix
||Predicted Negative (PN)|Predicted Positive (PP)|
|:-:|:-:|:-:|
|Actual Negative (AN)|True Negative (TP)|False Positive (FP)|
|Actual Positive (AP)|False Negative (FN)|True Positive (TP)|
### Accuracy
$Accuracy=\frac{TP+TN}{Total}$

Total number of labels predicted correctly as a proportion of the total number of predictions

Example:
||PN|PP|
|:-:|:-:|:-:|
|AN|998|0|
|AP|1|1|

$Accuracy=\frac{1+998}{1000}=0.999$
### Precision and Recall
#### Precision
$Precision=\frac{TP}{TP+FP}$

Accuracy of positive predictions/How accurate classifier is at predicting the correct class
#### Recall
$Recall=\frac{TP}{TP+FN}$

Ratio of positive instances that are correctly detected by the classifier/Sensitivity of classifier at detecting positive instances
#### Trade-off
F<sub>1</sub> score is the harmonic mean of Precision and Recall

# Supervised Learning: Classification and Regression with Non-Linear models (Decision Trees)

## Decision Tree
A tree of decisions and their possible consequences to show the decision-taking steps
### Types
#### Classification
Target variable is categorical/discrete (e.g. Yes/No)
#### Regression
Target variable is continuous (e.g. Income)
### Terminology
- Root node: Represents entire population or sample
- Splitting: Process of subdividing a node into sub-nodes
- Decision node: A sub-node which can be split into further sub-node
- Leaf/Terminal node: A sub-node which does not split
- Branch/Sub-tree: A sub-section of the entire tree
### Training (Building)
- Root node chosen by using attribute and split which results in the greatest reduction in node impurity
- Greedy approach: Top-down, recursive, divide and conquer
- Split data (node) based on selected attribute(s) by testing using an algorithm
  - Categorical attributes: Divide values into 2 subsets
  - Continuous attributes: Consider all possible splits and find best 2 subset
  - Nodes with a homogenous distribution are preferred, split in such a way as to minimise impurity
### Stopping
#### Predefined stopping condition
- All data for a given node belongs to the same class
- No remaining attributes for further splitting
- Number of observations per node is less than a predefined $x$
#### 100% Accuracy
- There is 1 leaf for each observation
- Overfitting has occured and the tree needs to be pruned with a constraint put on size
#### Ensembles
- Combine several trees to produce better predictive performance (collective intelligence)
- Bagging (Bootstrap aggregation) and boosting methods
