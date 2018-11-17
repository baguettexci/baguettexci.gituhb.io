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

```python

```
{% highlight text %}

{% endhighlight %} 
