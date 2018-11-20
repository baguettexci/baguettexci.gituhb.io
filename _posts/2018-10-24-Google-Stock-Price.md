---
title: Predicting Google Stock Price
date: 2018-10-24
tags: 
  - machine learning
  - data science
header:
  image: ""
excerpt: "Machine Learning, Data Science"
mathjax: "true"
---

Predicting some the upward and downward trends
Predicting the open google stock price, that is the stock price at the beginning of the financial day
It performs well better than the traditional ARIMA model


Train the model to be able to predict the stock price at time t+1, based on the previous 60 stock prices 
## 1) Setup
### Load packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

### Load & Viewing the Data
```python
# Load dataset
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv(filename, delim_whitespace=True, names=names)
dataset.head(10)
```



<img src="{{ site.url }}{{ site.baseurl }}/images/Google Stock Price/h.png" alt="">
```python

```
{% highlight text %}

{% endhighlight %} 
