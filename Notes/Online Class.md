# Class 1

**Machine Learning**: System is learning. Trying to understand relationship/pattern between data and produce the answer. So every output is a probability/probable answer.

Two types of machine learning:
1. **Supervised machine learning**: Here you already know how your output looks like. Here we have *input* and *output* in the data 
2. **Unsupervised machine learning**: Here we do clustering of related data(We can group based on our own way of classification).

Classification of Supervised Machine Learning:
1. **Regression**: Here we have a continuous kind of data
2. **Classification**: Here goal is to categorize data points into <u>predefined classes</u> or categories based on input features.

Not to be confused with `Classification` and `Clustering`

| Classification                        | Clustering                                                                         |
| ------------------------------------- | ---------------------------------------------------------------------------------- |
| Here classes are predefined and known | Here the number of nature of clusters are discovered during the clustering process |

---
# Class 2

For example  the `temperature` is a regression kind of output(continuous) and rolling a dice is a classification kind of output (*it has fixed 6 output values*)

Algorithms in `Regression` and `Classification`

| Regression                       | Classification                   |
| -------------------------------- | -------------------------------- |
| Linear Regression                | Logistic Regression              |
| Decision Tree                    | Decision Tree Classifier DTC     |
| Random Forest                    | Random Forest Classifier RFC     |
| KNN -> K-Nearest                 | KNN                              |
| SVR -> Support Vector Regression | SVC -> Support Vector Classifier |
Also in unsupervised learning we have some of these:

1. K Mean Clustering
2. Hierarchical Approach
3. DB Scan

#### Algorithms in supervised learning
1. **Linear Regression**:
	Works better with linear relationship(linear slope). Or simply we  have linear relationship between dependent and independent columns/data.
	1. Dependent Column <--> Output Column
	2. Independent Column <--> Input Column
	And with only one input column then it's a simple linear regression.  

| Experience | Salary |
| ---------- | ------ |
| 1          | 10000  |
| 2          | 20000  |
| 3          | 30000  |
| 4          | 40000  |
| 5          | 50000  |
The scatter plot of this would look like
![[Pasted image 20240613091255.png]]

Now the slope(β<sub>1</sub>) of the line can be determined by using the formula

$$
β_0=\frac{\sum (x - \bar{x})(y - \bar{y})}{\sum (x - \bar{x})^2}​
$$

Here the part of this formula is divided into 2 parts
1. Numerator: (**Covariance**) Represents the covariance between x and y, indicating how they change together.
2. Denominator: (**Variance**) Represents the variance of x, indicating how much x spreads around its mean.

**Cost Function**: This is error between actual value and predicted value or it is the measure of difference between predicted values produced by a model and the actual values from the dataset. It provides the measure of how "wrong" the models prediction are.

$$
f(β_0, β_1) = f(slope, y intercept(c))
$$

The picture shows the formula for MSE (first one) and MAE (second one).

![[Pasted image 20240613094750.png]]

So the formula to calculate cost function is:

1. **Mean Absolute Error (MAE)**
	Measures the absolute difference between actual and predicted values. Less sensitive than MSE.
$$
Mean Absolute Error: \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)
$$
	Where,
	- `n` - Number of samples in dataset
	- `y` - Expected value
	- `ŷ` - Predicted value
	- `i` - Index of sample

	Simple python code to calculate the MAE would be,
```python
def mae(predictions: list, targets: list):
sample_size = len(targets) # Assuming both list have same sample size
total_error = 0.0

for prediction, target in zip(predictions, target) # Create list of 2
total_error += abs(prediction - target)

mae_error = (1.0 / sample_size) * total_error

return mae_error
```

2. **Mean Squared Error (MSE)** 
	It is the average squared difference between the prediction and expected results. In case of MAE each error values were equal to the difference between the distance of coordinates but in case of MSE the partial error is equivalent to the area of the square created out of the geometrical distance between the measured points. And all regional areas are summed up and averaged.
![[Pasted image 20240613100753.png]]

MSE can be calculated by using the formula:
$$
Mean\;Squared\;Error (MSE) = \frac{1}{2m} \sum_{i=1}^m (Y_i - \hat{Y}_i)^2
$$

Where,
-  `n` - Number of samples in dataset
- `y` - Expected value
- `ŷ` - Predicted value
- `i` - Index of sample


Another formula for MSE is:
$$
Mean\;Squared\;Error = \frac{1}{2n} \sum_{i=1}^{n}(y_i - (m + c*x_i))^2

$$
$$
ŷ = m + c*x_i
$$

And using,

$$
f(β_0,\;β_1) = f(slope(m),\;y\;intercept(c)) = Cost\;Function (MSE)
$$

We get,

$$
Mean\;Squared\;Error = \frac{1}{2n}\sum_{i=1}^{n} (β_0 + β_1 * x_i)
$$




Now for reduction of error we can update the parameter. Here we are changing the value of m (slope of regression line) and c (y-intercept).
$$
m = \alpha\frac{dy}{dm}
$$

$$
c = \alpha\frac{dy}{dc}
$$

Where,
-  `α` - Learning Rate


(Yes the part from now on is part of pre-recorded session of class 2)

### Visualization of data using seaborn, pyplot and pandas:

**NOTE:** All the code will have the following import and is not included for redundancy.

For installation run the following commands:
```bash
pip install matplotlib seaborn pandas numpy
```

**IMPORT:**
```python
import matplotlib.pyplot as plot
import seaborn as sns
import pandas as pd
import numpy as np
```

Simple pair-plot:
```python
data_frame=pd.read_csv('Advertising.csv')
sns.pairplot(data_frame, x_vars=["TV", "Radio", "Newspaper"], y_vars=["Sales"])
plot.show() # This one is optional if you are in Jupyter notebook
```

OUTPUT:
![[Pasted image 20240613112804.png]]



Here we can notice that the TV vs Sales have a more linear kind of relationship as compared to others. Furthermore we can show the multicollinearity between the input data/columns (correlation >= 0.9) using the heat-map plot. (More about multicollinearity in Class 4 section)

```python
sns.heatmap(data_frame.corr(), annot=True)
plot.show() # Optional for jupyter notebook
```

OUTPUT:
![[Pasted image 20240613113107.png]]

Here we can observe that there is no multicollinearity between the **input columns** since all of them are <0.9 and note that sales is a output column so you can't consider that. If we set `annot=False` in `Seaborn.heatmap()` then the resulting plot will not show the correlation inside individual box. 

### Splitting data for training and testing

Here we split out data set into 20% and 80%. The 80% of data we use for training out model and rest 20% to test the model. This one requires a special import too which will help in splitting our data into required portion defined in `test_size=n`

```python
from sklearn.model_selection import train_test_split
data_frame = pd.read_csv('Advertising.csv')
x = data_frame.drop("Sales", axis=1) # Remove Sales column and keep rest
y = data_frame['Sales']              # Only take Sales column
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

Here one thing is to note is that the 80-20 division is not like first 80 and last 20 from file but is totally random and this is controlled by `random_state=n`. By specifying this parameter the function will generate the same split of data for same value of `n` each time the code is ran.

**Fun fact:**

> The use of the number 42 as the **random_state** parameter in machine learning is actually a reference to the science fiction series "The Hitchhiker's Guide to the Galaxy" by Douglas Adams. In the series, the number 42 is famously referred to as the "Answer to the Ultimate Question of Life, the Universe, and Everything," though the actual question is unknown. It's a humorous and whimsical reference that has been embraced by the programming and data science community.


### Using linear regression library with split data


```python
import skearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
lr.intercept_ # will give the value of y-intercept(c)
lr.coef_ # Regression Coefficient (slope/m)

predicted = lr.predict(x_test) # Now you can test this against y_test
actual = y_test

import sklearn.metrics import mean_squared_error

error = mean_squared_error(predicted, actual) # Now we get here our mean squared error
```

---

# Class 3

### Algorithms in Classification kind of problem

1. **Logistic Regression**: Also called as **sigmoid function** or **logistic function**. The formula for this function is
	$$
	P(x) = \frac{1}{1+e^{-(mx+c)}}
	$$
	This function converts the data in range between 0 to 1. So if it's towards 1 it will be in class 1 and if its near 0 then its class 0. Here are only 2 classes for binary classification. And for other number of classes we have different algorithms. This a sigmoid function and it always intersects y-axis at 0.5.
	
	![[Pasted image 20240614103518.png]]

	And in cases of more than 2 classes, we use the approach One vs All `-or-` One vs Rest.
	![[Pasted image 20240614104602.png]]

###### Accuracy Metrices:

Some classification of accuracy metrics are:
1. **Accuracy**
	 Before calculating accuracy we need to understand `True/False` `Positive/Negative`
	 - If both result are same then its `True` otherwise `False`
	 - If its `0` then its negative and for `1` its positive. So combine them together we get,

| Predicted \Actual | 0              | 1              |
| ----------------- | -------------- | -------------- |
| 0                 | True Negative  | False Negative |
| 1                 | False Positive | True Positive  |
	Fig: Confusion Matrix

Calculating accuracy is pretty easy. For example we have 100 data items distributed as the following

| Predicted/Actual | 0   | 1   |
| ---------------- | --- | --- |
| 0                | 30  | 21  |
| 1                | 27  | 31  |
Now the formula for accuracy is:
$$
Accuracy = \frac{TP + TN}{Total\;Items}
$$
Where,
- `TP` - True Positive
- `TN` - True Negative
So calculating the above table using this formula we get,
$$
Accuracy = \frac{30+31}{30+31+21+27} = 0.61 = 61\%
$$


2. **Recall**: 
	Gives how sensitive the model is / Gives how much the model as predicted positives. It is basically a quantity of positive. It doesn't considers the false positive. Recall is a fundamental concept in machine learning that measures the proportion of relevant instances that are correctly identified by a model out of all the instances that are actually relevant. It is also known as the true positive rate (TPR) or sensitivity.
	The formula to calculate Recall is:
	$$
	Recall = \frac{TP}{TP + FN}
	$$
	Using this formula in table we get the recall as `52%`.


3. **Precision**
	Gives quality of positives/ How much is actual positives. The formula for calculating precision is:
	$$
	Precision = \frac{TP}{TP + FP}
	$$
	For the above table we get precision as `71%` which means from 52% (accuracy) the 71% are actually positives.<br>
4. **F1 -Score:** 
	The F1 score is a measure of the harmonic mean of precision and recall. It is commonly used in machine learning to evaluate the performance of a classification model, particularly in cases where the positive class is rare or class imbalance exists. <br>The formula to calculate the F1-Score is,
	$$
	F1\;Score = \frac{2(Precision * Recall)}{Precision + Recall}
	$$
	Using this formula for above table we get F1-Score as `61%`. And if its somewhere middle then we are going right.

---
# Class 4 + 5
### Correlation

Correlation is the finding of degree of relation between 2 variables.
- -1 ---> No relation (Totally opposite to each other)
-  0 ---> Neutral relation
-  1 ---> 100% related (Both data are same)

#### Multicollinearity:

If the correlation between 2 <u>input</u> variables is >= 0.9 then it refers to multicollinearity.(Only the input variables). There is no multicollinearity  between output columns.


2. **Decision Tree Algorithm:**
	Decision tree is like a tree from data structures in which every node is classified into multiple classes and this classification is clearly specified by Gini i.e which will become root/child, etc.
	1. **Gini:** It helps to decide which column becomes the root column. It is a measurement of incorrect classification of data. Formula to calculate Gini is given by,
		$$
		Gini = 1- \sum_{i=1}^{n}{(P_i)^2}
		$$
		Here i is number of classes we have with target column. Lets take this example:

| Class | Yes | No  | Total |
| ----- | --- | --- | ----- |
| 8     | 2   | 0   | 2     |
| 9     | 1   | 1   | 2     |
| 10    | 1   | 1   | 2     |
| 11    | 0   | 1   | 1     |
	Now we can find out the probability of yes/no for each class. Below is the table for living in hostel or not,

| Class | P(Yes) | P(No) |
| ----- | ------ | ----- |
| 8     | 1      | 0     |
| 9     | 0.5    | 0.5   |
| 10    | 0.5    | 0.5   |
| 11    | 0      | 1     |
Now if we want to find Gini of class 8 then it would be calculated as,
$$
G(8) = G(10) = 1 - P(Yes)^2 - P(No)^2 = 1 - 1^2 - 0^2 - 0
$$
$$
G(9) = G(10) = 0.5
$$
Now for finding Gini of a column class is,
$$
G(Class) = \sum_{i=8}^{11}\frac{Number\;of\;instances}{Total\;Instances}\;* G(i)
$$

This formula is specially for the above table ranging from 8 to 11. And using this formula for above table we get `G(Class) = 0.29. Now lets calculate `Gini(Gender)`,

| Gender | P(Yes) | P(No) |
| ------ | ------ | ----- |
| Male   | 0.5    | 0.5   |
| Female | 3/5    | 2/5   |
Now the `Gini(Male) = 0.5` and `Gini(Female) = 0.48` and `G(Gender) = 0.49` and we got previously `G(Class) = 0.29` now this value will help us to decide which column to use as the root node/column in tree. **Higher the Gini, higher will be the impurity. So the Gini data specifies if any new data comes it can be misclassified.** So here in this example we will choose `Class` as the root column.
![[Pasted image 20240616131436.png]]
Finding the upper and lower limit of dataset:
- Upper Limit = `Q3 + 1.5 * IQR`
- Lower Limit = `Q1 - 1.5 * IQR`
Calculation of Quartiles is given by:
- Q1 - `(n+1)/4`
- Q2 - `(n+2)/2`
- Q3 - `3(n+1)/4`
Calculation of IQR(*Inter-Quartile Range*) is given by:
- IQR = `Q3 - Q1`

Now lets consider our dataset as `[1, 2, 3, 5, 7, 9, 10, 11, 53]`. Calculating the upper and lower limit we get UL = 92 and LL = 4 so anything outside this will be removed from dataset. Thse are called **Outliers**. Outliers are unimportant data that are to be removed from the dataset. Using python thiss can be done as:
```python
import numpy as np
import pandas as pd

data_frame = pd.read_csv("dataset.csv");
column_name = (open("dataset.csv").readLine()).split(",")

for cols in column_name:
	q1 = data_frame[col].quantile(0.25)
	q3 = data_frame[col].quantile(0.75)
	iqr = q3 - q1
	upper_limit = q3 + iqr * 1.5
	lower_limit = q1 - iqr * 1.5
	data_frame[col] = np.clip(data_frame[col], upper_limit, lower_limit)
```

---
# Class 6

### Information Gain:

Information gain is calculated as `IG = Entropy(before) - Entropy(after)`. And for calculation of entropy,
$$
Entropy = -\sum_{i=1}^{n}P_i\;log_2(P_i)
$$
- Before -> Target column
- After -> Input column
- `P` -> Probability

Furthermore,
- Low Entropy = High IG (SO this will be our class column or root column)
- High Entropy = Low IG

# Class 7

### KNN - K Nearest Neighbour Algorithm
 - Lazy Algorithm
 - For supervised ML
 - Simple way to classify data
 - K = Kth neighbour
 Calculating is done by,
 $$
 K-Nearest=\sqrt{(x_1-x_2)^2 + (y_1-y_2)^2)}
 $$
### K-Mean clustering
- For unsupervised ML
- k = How many groups are required to classify data
Steps for dividing datapoints in cluster involve the following steps:
1. Assume k=`2` meaning 2 grpups of data points will be there
2. Select randomly 2 datapoints (this is out first centeroid)
3. Pick nearest datapoints to centeroid
4. Centeroid will change since the group expands for each 2 clusters parallelly
5. And this continues untill all data points are covered.

Note that the clusters don't overlap with each other. Now for understanding how we can get the value of k we need to understand few of the more terms:
- **WCSS** - Within Cluster Summation Square
	Assume that we have `n` number of datapoints and `k` number of cluster, then for calculation of WCSS we first need to calculate Sum of Square (SS)
	$$
	SS_1 = \sum_{i=1}^{n_1}(C_1 - P_i)
	$$
	$$
	SS_k = \sum_{i=1}^{n_2}(C_k-P_i)
	$$
	and for the WCSS it will be,
	$$
	WCSS = \sum_{i=1}^{k}{SS_k}
	$$
	When k = n then `WCSS = 0` and when k = 1 `WCSS = MAX` and for any other value of `k` the WCSS will be $0 < WCSS < MAX$ 
### Albo Graph

![[Pasted image 20240617194856.png]]

Using albo graph we decide the value of `K` value and the plotting is against `WCSS` vs `N`

Buttttt there's still some problem of choosing random 2 centroids. Randomly choosing centroids will cause biased distribution of data so we have k-means clustering ++ method, an upgraded clustering method.

SO how to solve it?
- You can choose first centroid randomly
- Second also choose randomly

Now you can calculate distance between these 2 centroids using formula $Z^2=X^2-Y^2$ . If this is $<= Mean$ then you can choose this centroid else skip this centroid


# Class 9 + 10


1. Calculate mean distance to points in same cluster and for all cluster.