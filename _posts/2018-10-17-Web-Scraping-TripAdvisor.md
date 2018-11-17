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

The data would then be contained in a csv.
<br/>
The items to be retrieved are:
* Content of the reviews
* Ratings of the reviews
* Reviewed date

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
      <th>16677</th>
      <td>3</td>
      <td>I stayed with my partner in May. For the price...</td>
      <td>June 7, 2010</td>
    </tr>
    <tr>
      <th>16678</th>
      <td>4</td>
      <td>Stayed there with my family, spacious, modern ...</td>
      <td>June 2, 2010</td>
    </tr>
    <tr>
      <th>16679</th>
      <td>1</td>
      <td>I would avoid the Marina Bay Sands, at least f...</td>
      <td>May 14, 2010</td>
    </tr>
    <tr>
      <th>16680</th>
      <td>1</td>
      <td>I stayed here for four days for a conference a...</td>
      <td>May 13, 2010</td>
    </tr>
    <tr>
      <th>16681</th>
      <td>1</td>
      <td>I stayed 6 nights at this hotel recently, havi...</td>
      <td>May 12, 2010</td>
    </tr>
  </tbody>
</table>

### Change the datetime format
```python
mbs['review_date'] = pd.to_datetime(mbs['review_date'])
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
      <td>2018-11-07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>Yes it’s huge, yes it’s expensive but honestly...</td>
      <td>2018-11-07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>Iconic hotel very good public areas staff are ...</td>
      <td>2018-11-07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>We stayed 1 night in this hotel because I want...</td>
      <td>2018-11-07</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Very helpful staff during checkin. Room was cl...</td>
      <td>2018-11-07</td>
    </tr>
  </tbody>
</table>

## 2) Exploratory data analysis

```python
sns.countplot(y=mbs['rating'], color='mediumseagreen', order=[5,4,3,2,1])
```
<img src="{{ site.url }}{{ site.baseurl }}/images/WebScrapingTripAdvisor/countplot.png" alt="">

```python
sns.countplot(x = 'year', data=mbs, palette="GnBu_d")
```
<img src="{{ site.url }}{{ site.baseurl }}/images/WebScrapingTripAdvisor/countplot2.png" alt="">

```python
mbs['year'] = mbs['review_date'].dt.year
```
```python
plt.figure(figsize=(12,8)) 
ax= sns.countplot(x='year' ,hue='rating',data=mbs, palette="Set3")
ax.set(xlabel='Year', ylabel='Count')
ax.figure.suptitle("Ratings by Year", fontsize = 20)
plt.show()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/WebScrapingTripAdvisor/countplot3.png" alt="">

```python
import string
from nltk.corpus import stopwords

def text_process(text):
    # Remove any punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove any stopwords
    # NLTK's stopwords assumes words are all lowercased
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]

    #Join the characters again to form the string
    return " ".join(text)
```
```python
reviews = mbs['review_body'].apply(text_process)
reviews.head()
```
{% highlight text %}
0    stayed one night hereI decide upgrade Club55 a...
1    Yes it’s huge yes it’s expensive honestly woul...
2    Iconic hotel good public areas staff amazing l...
3    stayed 1 night hotel wanted swim rooftop pool ...
4    helpful staff checkin Room clean bright Given ...
Name: review_body, dtype: object
{% endhighlight %} 

```python
from sklearn.feature_extraction.text import CountVectorizer
# Create a bag of words feature matrix
count = CountVectorizer()
bag_of_words = count.fit_transform(reviews)
```

```python
import collections

word_freq = dict(zip(count.get_feature_names(), np.asarray(bag_of_words.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(30), columns = ['word', 'freq'])

fig, ax = plt.subplots(figsize=(12, 10))
#sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
sns.barplot(x="freq", y="word", data=word_counter_df, palette="Blues_d", ax=ax, orient="h")
plt.grid(linewidth=1, alpha=0.3, color='lightgrey')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.title('Most Common 30 Words')
plt.show()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/WebScrapingTripAdvisor/barplot.png" alt="">

## 3) N-grams
```python
from nltk.util import ngrams
## Helper Functions
def get_ngrams(text, n):
    n_grams = ngrams((text), n)
    return [ ' '.join(grams) for grams in n_grams]
```
```python
from collections import Counter
def gramfreq(text,n,num):
    # Extracting bigrams
    result = get_ngrams(text,n)
    # Counting bigrams
    result_count = Counter(result)
    # Converting to the result to a data frame
    df = pd.DataFrame.from_dict(result_count, orient='index')
    df = df.rename(columns={'index':'words', 0:'frequency'}) # Renaming index column name
    return df.sort_values(["frequency"],ascending=[0])[:num]
```
```python
def gram_table(text, gram, length):
    out = pd.DataFrame(index=None)
    for i in gram:
        table = pd.DataFrame(gramfreq(preprocessing(text),i,length).reset_index())
        #table.columns = ["{}-Gram".format(i),"Occurence"]
        #out = pd.concat([out, table], axis=1)
        #table = pd.DataFrame(gramfreq(x,i,length).reset_index())
        table.columns = ["{}-Gram".format(i),"Frequency"]
        out = pd.concat([out, table], axis=1)
    return out
```
```python
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
stop_words = set(stopwords.words('english'))
def preprocessing(data):
    txt = data.str.lower().str.cat(sep=' ') #1
    #txt = txt.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(txt)
    words = [w for w in words if not w in stop_words] #3
    porter = PorterStemmer()
    words = [porter.stem(word) for word in words]
    return words
```
```python
gram_table(reviews, gram=[1,2,3,4], length=15)
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1-Gram</th>
      <th>Frequency</th>
      <th>2-Gram</th>
      <th>Frequency</th>
      <th>3-Gram</th>
      <th>Frequency</th>
      <th>4-Gram</th>
      <th>Frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hotel</td>
      <td>31203</td>
      <td>marina bay</td>
      <td>5649</td>
      <td>marina bay sand</td>
      <td>4216</td>
      <td>stay marina bay sand</td>
      <td>1057</td>
    </tr>
    <tr>
      <th>1</th>
      <td>room</td>
      <td>30514</td>
      <td>infin pool</td>
      <td>5315</td>
      <td>stay marina bay</td>
      <td>1167</td>
      <td>marina bay sand hotel</td>
      <td>474</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pool</td>
      <td>21737</td>
      <td>bay sand</td>
      <td>4327</td>
      <td>5 star hotel</td>
      <td>664</td>
      <td>infin pool 57th floor</td>
      <td>211</td>
    </tr>
    <tr>
      <th>3</th>
      <td>stay</td>
      <td>18657</td>
      <td>garden bay</td>
      <td>2066</td>
      <td>citi view room</td>
      <td>508</td>
      <td>night marina bay sand</td>
      <td>203</td>
    </tr>
    <tr>
      <th>4</th>
      <td>view</td>
      <td>16684</td>
      <td>citi view</td>
      <td>1959</td>
      <td>stay one night</td>
      <td>491</td>
      <td>marina bay sand one</td>
      <td>118</td>
    </tr>
    <tr>
      <th>5</th>
      <td>servic</td>
      <td>9289</td>
      <td>club room</td>
      <td>1801</td>
      <td>bay sand hotel</td>
      <td>485</td>
      <td>visit marina bay sand</td>
      <td>91</td>
    </tr>
    <tr>
      <th>6</th>
      <td>check</td>
      <td>9209</td>
      <td>swim pool</td>
      <td>1753</td>
      <td>view garden bay</td>
      <td>434</td>
      <td>singapor marina bay sand</td>
      <td>88</td>
    </tr>
    <tr>
      <th>7</th>
      <td>singapor</td>
      <td>9081</td>
      <td>stay hotel</td>
      <td>1651</td>
      <td>pool 57th floor</td>
      <td>414</td>
      <td>club room citi view</td>
      <td>85</td>
    </tr>
    <tr>
      <th>8</th>
      <td>night</td>
      <td>9021</td>
      <td>view room</td>
      <td>1462</td>
      <td>room citi view</td>
      <td>356</td>
      <td>marina bay sand singapor</td>
      <td>84</td>
    </tr>
    <tr>
      <th>9</th>
      <td>bay</td>
      <td>9000</td>
      <td>one night</td>
      <td>1421</td>
      <td>ku de ta</td>
      <td>342</td>
      <td>experi marina bay sand</td>
      <td>76</td>
    </tr>
    <tr>
      <th>10</th>
      <td>one</td>
      <td>8930</td>
      <td>5 star</td>
      <td>1259</td>
      <td>stay 2 night</td>
      <td>283</td>
      <td>marina bay sand stay</td>
      <td>73</td>
    </tr>
    <tr>
      <th>11</th>
      <td>great</td>
      <td>8406</td>
      <td>stay marina</td>
      <td>1208</td>
      <td>infin pool amaz</td>
      <td>273</td>
      <td>hotel marina bay sand</td>
      <td>69</td>
    </tr>
    <tr>
      <th>12</th>
      <td>time</td>
      <td>8078</td>
      <td>57th floor</td>
      <td>1188</td>
      <td>roof top pool</td>
      <td>259</td>
      <td>roof top infin pool</td>
      <td>63</td>
    </tr>
    <tr>
      <th>13</th>
      <td>staff</td>
      <td>7921</td>
      <td>shop mall</td>
      <td>1131</td>
      <td>rooftop infin pool</td>
      <td>254</td>
      <td>room face garden bay</td>
      <td>63</td>
    </tr>
    <tr>
      <th>14</th>
      <td>get</td>
      <td>7783</td>
      <td>great view</td>
      <td>1125</td>
      <td>infin pool 57th</td>
      <td>239</td>
      <td>swim pool 57th floor</td>
      <td>57</td>
    </tr>
  </tbody>
</table>

## 4) Topic Modeling with Latent Dirichlet Allocation
```python
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_topics=20, learning_method="batch", max_iter=25, random_state=0)
# Build the model and transform the data in one step
# Computing transform takes some time, and we can save time by doing both at once.
document_topics = lda.fit_transform(bag_of_words)
```
```python
lda.components_.shape
```
{% highlight text %}
(20, 51339)
{% endhighlight %} 

```python
# for each topic (a row in the components_), sort the features (ascending).
# Invert rows with [:, ::-1] to make sorting descending
sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
# Get the feature names from the vectorizer:
feature_names = np.array(count.get_feature_names())
```
```python
def print_topics(topics, feature_names, sorting, topics_per_chunk, n_words):
    for i in range(0, len(topics), topics_per_chunk):
        # for each chunk:
        these_topics = topics[i: i + topics_per_chunk]
        # maybe we have less than topics_per_chunk left
        len_this_chunk = len(these_topics)
        # print topic headers
        print(("topic {:<8}" * len_this_chunk).format(*these_topics))
        print(("-------- {0:<5}" * len_this_chunk).format(""))
        # print top n_words frequent words
        for i in range(n_words):
            try:
                print(("{:<14}" * len_this_chunk).format(
                    *feature_names[sorting[these_topics, i]]))
            except:
                pass
        print("\n")
```
```python
# print out the 20 topics:
print_topics(topics=range(20), feature_names=feature_names, sorting=sorting, topics_per_chunk=6, n_words=10)
```
{% highlight text %}
topic 0       topic 1       topic 2       topic 3       topic 4       topic 5       
--------      --------      --------      --------      --------      --------      
room          balcony       kids          spore         hotel         marina        
pool          quite         even          hv            pool          bay           
view          doorstep      nice          skylark       singapore     sands         
amazing       deluxe        much          simply        bay           singapore     
great         inner         food          60th          view          stay          
night         pricy         chinese       creation      great         time          
check         chill         race          dax           marina        staff         
staff         room          place         minus         amazing       like          
stay          bayfront      family        prize         infinity      hotel         
us            cctv          come          rises         city          room          


topic 6       topic 7       topic 8       topic 9       topic 10      topic 11      
--------      --------      --------      --------      --------      --------      
casino        club          check         room          go            service       
mall          room          universal     hotel         dollars       dress         
diverse       suite         studios       would         even          code          
roads         floor         minute        mbs           floor         guest         
hotel         view          free          size          shangri       industry      
october       lounge        pleasant      service       sing          management    
frontdesk     city          cavalli       bed           smokers       looks         
notice        staff         roberto       stayed        passport      staffs        
luxury        breakfast     things        large         la            cannot        
ie            pool          different     well          charge        ppl           


topic 12      topic 13      topic 14      topic 15      topic 16      topic 17      
--------      --------      --------      --------      --------      --------      
birthday      der           room          hotel         baggages      room          
mbs           ist           view          pool          coupon        hotel         
stay          die           pool          stay          son           check         
cake          sehr          bathroom      like          blazer        us            
us            und           bay           people        capsules      service       
anniversary   das           nice          rooms         old           staff         
thank         ein           hotel         service       christmas     one           
service       mbs           also          room          uncle         would         
special       laundry       infinity      view          arround       told          
team          pad           bath          one           vip           get           


topic 18      topic 19      
--------      --------      
room          service       
hotel         offered       
also          reward        
pool          bond          
food          teh           
get           theatre       
area          venue         
floor         it            
good          399           
tower         jacuzzi           
{% endhighlight %} 

```python
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
topic_names = ["{:>2} ".format(i) + " ".join(words) for i, words in enumerate(feature_names[sorting[:, :2]])]

# two column bar chart:
for col in [0, 1]:
    start = col * 10
    end = (col + 1) * 10
    ax[col].barh(np.arange(10), np.sum(document_topics, axis=0)[start:end])
    ax[col].set_yticks(np.arange(10))
    ax[col].set_yticklabels(topic_names[start:end], ha="left", va="top")
    ax[col].invert_yaxis()
    ax[col].set_xlim(0, 5000)
    yax = ax[col].get_yaxis()
    yax.set_tick_params(pad=130)
plt.tight_layout()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/WebScrapingTripAdvisor/chart.png" alt="">




```python

```
{% highlight text %}

{% endhighlight %} 
