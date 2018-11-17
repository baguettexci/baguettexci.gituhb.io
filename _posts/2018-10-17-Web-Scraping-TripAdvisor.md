---
title: Web Scraping TripAdvisor and Text Mining
date: 2018-10-17
tags: 
  - Web Scraping
  - NLP
header:
  image: ""
excerpt: "Web Scraping, NLP"
mathjax: "true"
---
As lots of unstructured data is available on the internet, the dataset used will be extracted from TripAdvisor. 
TripAdvisor is a widely known website that shows hotel and restaurant reviews, accommodation bookings and other travel-related content.
I performed a web scraping on TripAdvisor for the hotel, Marina Bay Sands, using [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) and an example from [furas](https://github.com/furas).




<img src="{{ site.url }}{{ site.baseurl }}/images/WebScrapingTripAdvisor/page1.png" alt="">

The items to be retrieved are:
* Content of the reviews
* Ratings of the reviews
* Reviewed date
<br/>
The data would then be contained in a csv.

<img src="{{ site.url }}{{ site.baseurl }}/images/WebScrapingTripAdvisor/page2.png" alt="">

## 1) Setup
### Load packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### Load & Viewing the Data
```python
mbs = pd.read_csv('mbs.csv', sep=',')
mbs.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>review_body</th>
      <th>review_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>I stayed for one night here.I decide to upgrad...</td>
      <td>November 7, 2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>Yes it’s huge, yes it’s expensive but honestly...</td>
      <td>November 7, 2018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>Iconic hotel very good public areas staff are ...</td>
      <td>November 7, 2018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>We stayed 1 night in this hotel because I want...</td>
      <td>November 7, 2018</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Very helpful staff during checkin. Room was cl...</td>
      <td>November 7, 2018</td>
    </tr>
  </tbody>
</table>

```python
mbs.tail()
```


```python

```
{% highlight text %}

{% endhighlight %} 
