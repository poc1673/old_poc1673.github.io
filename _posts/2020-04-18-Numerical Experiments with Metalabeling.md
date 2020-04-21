# Numerical Experiment with Metalabeling

# What is Metalabeling?

Metalabeling is a secondary machine learning model built on the results of the primary classification model to identify and reclassify false positives. It was first suggested in [Advances in Financial Machine Learning](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos-ebook/dp/B079KLDW21/ref=sr_1_3?crid=2YD9Z0HQ4TRR0&dchild=1&keywords=advances+in+financial+machine+learning&qid=1587241148&sprefix=advances+in+machine%2Caps%2C207&sr=8-3) by Marcos López de Prado and given the relative novelty of the method (the book was published in 2018), I wanted to do a short test.

To provide more detail: When we perform metalabeling we fit a primary classification model on the data to classify the results based on an initial set of parameters. Once the model is fit, the we compare the results to the predictions and divides them into two groups (the metalabels): 0 if the observation is a true positive, 1 if it is a false positive.

These two groups are then used to fit a **secondary model** which detects the false positives. This secondary model is used to filter the results. 

# The Banknote Data-set

As a quick test of the combined primary and secondary models, I used a bank note data-set available from [UCI](https://archive.ics.uci.edu/ml/datasets/banknote+authentication). This dataset contains 1,372 observations of four input variables to predict the authenticity of banknotes. A result of 0 indicates authenticity while 1 indicates a fraudulent note.


```python
import pandas as pd
dat = pd.read_csv("C:\\Users\\USER\\Dropbox\\Projects\\Metalabeling Exercise\\data_banknote_authentication.txt",header = None)
dat.columns = ["Variance","Skewness","Kurtosis","Entropy","Class"]
dat.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variance</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
      <th>Entropy</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.62160</td>
      <td>8.6661</td>
      <td>-2.8073</td>
      <td>-0.44699</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.54590</td>
      <td>8.1674</td>
      <td>-2.4586</td>
      <td>-1.46210</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.86600</td>
      <td>-2.6383</td>
      <td>1.9242</td>
      <td>0.10645</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.45660</td>
      <td>9.5228</td>
      <td>-4.0112</td>
      <td>-3.59440</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.32924</td>
      <td>-4.4552</td>
      <td>4.5718</td>
      <td>-0.98880</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# Fitting the Classifier

The main purpose of this numerical example is to assess the improvement that metalabeling yields over using only a primary model. As a result, I will aim to use a simple binary logistic regression to categorize the results.


```python
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import numpy as np
from sklearn import tree
feature_data = dat[["Variance" ,"Skewness","Kurtosis","Entropy"]] 
transformer = Normalizer().fit(feature_data)  
normalized_data = transformer.transform(feature_data)
X_train, X_test, y_train, y_test = train_test_split( normalized_data, dat["Class"] , test_size=0.33, random_state=42)
clf = linear_model.LogisticRegression(random_state=13).fit(X_train, y_train)
hold = pd.DataFrame(y_train)
pd.value_counts(hold.Class)
pred_train =clf.predict(X_train)
pred_test =clf.predict(X_test)
train_cf = metrics.confusion_matrix(y_train,pred_train,labels = [1,0])
test_cf = metrics.confusion_matrix(y_test,pred_test,labels = [1,0])
```


```python
train_cf
```




    array([[411,   3],
           [ 10, 495]], dtype=int64)




```python
test_cf
```




    array([[193,   3],
           [  4, 253]], dtype=int64)



# Fitting the Secondary Model

In order to create the metalabeling function, I will use a simple decision tree as the secondary model. The function returning the secondary model is shown below and takes the following steps:

1. Identifies all positive predictions which were classified as positive.
2. Splits the results into true positives (0) and false positives (1).
3. Trains a decision tree model on the original training data on the metalabels from Step 2.
4. Identifies the false positives

I've combined the process of calculating the primary model (logistic regression) and the secondary model (decision tree) as one step. The implementation for this is provided below. The *model_w_metalabeling* function returns the primary model, the model with metalabeling, the training data and the testing data.


```python
def create_metalabeling(dep_vars, y_true, y_pred):
    pred_train_results = y_pred==1
    X_meta = dep_vars[pred_train_results]
    y_meta = y_true[pred_train_results]
    meta_labels = y_meta.apply(func= lambda x: 0 if x == 1 else 1   )    
    secondary_model = tree.DecisionTreeClassifier()
    secondary_results = secondary_model.fit(X_meta, meta_labels)
    secondary_predictions = pd.DataFrame(  {"Results" :secondary_model.predict(X_meta)}) 
    return({"Model": secondary_results,
            "Secondary_Predictions":secondary_predictions,
            "Confusion_Matrix" : metrics.confusion_matrix(secondary_predictions,meta_labels)})

def model_w_metalabeling(df_features,df_target,portion = .66):
    X_train, X_test, y_train, y_test = train_test_split( df_features, df_target , test_size=1-portion, random_state=13)
    primary_model = linear_model.LogisticRegression(random_state=13).fit(X_train, y_train)
    predicted_train = pd.DataFrame({"Results":primary_model.predict(X_train)})
    testing = create_metalabeling(X_train,y_train,primary_model.predict(X_train))
    secondary_model = testing["Model"]
    secondary_predictions = pd.DataFrame( {"Results" : secondary_model.predict(X_train[predicted_train.Results==1])
                                      } ).Results.apply(lambda x: 1 if x ==0 else 0)
    secondary_predictions.index = predicted_train[predicted_train.Results==1].index
    final_results  = pd.DataFrame({"Primary" : predicted_train.Results,
                                   "Secondary" : secondary_predictions}   )
    final_results["Final"] = np.where( (final_results.Primary == final_results.Secondary )|( final_results.Secondary.isna()), final_results.Primary, final_results.Secondary)
    
    def return_function(features,results):
        predicted_train = pd.DataFrame({"Results":primary_model.predict(features)})
        secondary_predictions = pd.DataFrame( {"Results" : secondary_model.predict(features[predicted_train.Results==1])
                                          } ).Results.apply(lambda x: 1 if x ==0 else 0)
        secondary_predictions.index = predicted_train[predicted_train.Results==1].index
        final_results  = pd.DataFrame({"Primary" : predicted_train.Results,
                                       "Secondary" : secondary_predictions}   )
        final_results["Final"] = np.where( (final_results.Primary == final_results.Secondary )|( final_results.Secondary.isna()), 
                                          final_results.Primary, final_results.Secondary)
        return(final_results)

    return({"model_w_meta":return_function,
            "primary_model":primary_model,
            "Training_Data": {"features":X_train,
                              "target":y_train},
           "Testing_Data": {"features":X_test,
                              "target":y_test}} )
```

## Testing the Results

For this example, I used a split with 80% training data and 20% testing data.


```python
testing_secondary_model = model_w_metalabeling(normalized_data, a["Class"],portion = .8)
meta_model = testing_secondary_model["model_w_meta"]
prim_model = testing_secondary_model["primary_model"]
X_train, y_train = [testing_secondary_model["Training_Data"]["features"],testing_secondary_model["Training_Data"]["target"] ]
X_test, y_test = [testing_secondary_model["Testing_Data"]["features"],testing_secondary_model["Testing_Data"]["target"]]
```

First, let's look at the results using the primary model:


```python
metrics.confusion_matrix(prim_model.predict(X_train),y_train,[1,0])
```




    array([[487,  10],
           [  3, 597]], dtype=int64)




```python
metrics.confusion_matrix(prim_model.predict(X_test),y_test,[1,0])
```




    array([[118,   4],
           [  2, 151]], dtype=int64)



We can see that it performs decently overall but a few false-positives are present (the upper right-hand corner of the confusion matrices above). 

The results from the model (metalabeling) models are shown below:


```python
metrics.confusion_matrix(meta_model(X_train,y_train).Final,y_train,[1,0])
```




    array([[487,   0],
           [  3, 607]], dtype=int64)




```python
metrics.confusion_matrix(meta_model(X_test,y_test).Final,y_test,[1,0])
```




    array([[118,   1],
           [  2, 154]], dtype=int64)



The in-sample results are as-expected. The use of the secondary model removes all false-positives which is what we would expect in-sample. The real test is the out-of-sample results which that three of the four false-positives from the primary model were mapped to the true-negative quadrant.

# Summary

Metalabeling offers a useful and simple tool for improving the effectiveness of machine learning models by training a secondary model to audit the positive results of the original model. Taken on face value, it's a very simple example of ensemble modeling. On a personal level, I find the simplicity in justifying and explaining this tool to an outsider. However, using this strategy places more weight on outcomes analysis since by forming this second model, it raises the likelihood of overfitting.
