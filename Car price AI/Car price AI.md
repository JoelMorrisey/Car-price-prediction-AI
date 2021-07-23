
  
# Car price prediction AI

### Attribute Information

1. symboling: -3, -2, -1, 0, 1, 2, 3. 
2. normalized-losses: continuous from 65 to 256. 
3. make: alfa-romero, audi, bmw, chevrolet, dodge, honda,isuzu, jaguar, mazda, mercedes-benz, mercury, mitsubishi, nissan, peugot, plymouth, porsche, renault, saab, subaru, toyota, volkswagen, volvo 
4. fuel-type: diesel, gas. 
5. aspiration: std, turbo. 
6. num-of-doors: four, two. 
7. body-style: hardtop, wagon, sedan, hatchback, convertible. 
8. drive-wheels: 4wd, fwd, rwd. 
9. engine-location: front, rear. 
10. wheel-base: continuous from 86.6 to 120.9. 
11. length: continuous from 141.1 to 208.1. 
12. width: continuous from 60.3 to 72.3. 
13. height: continuous from 47.8 to 59.8. 
14. curb-weight: continuous from 1488 to 4066. 
15. engine-type: dohc, dohcv, l, ohc, ohcf, ohcv, rotor. 
16. num-of-cylinders: eight, five, four, six, three, twelve, two. 
17. engine-size: continuous from 61 to 326. 
18. fuel-system: 1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi. 
19. bore: continuous from 2.54 to 3.94. 
20. stroke: continuous from 2.07 to 4.17. 
21. compression-ratio: continuous from 7 to 23. 
22. horsepower: continuous from 48 to 288. 
23. peak-rpm: continuous from 4150 to 6600. 
24. city-mpg: continuous from 13 to 49. 
25. highway-mpg: continuous from 16 to 54. 
26. price: continuous from 5118 to 45400.

## Data Exploration

###  Import Data


```python
import pandas as pd
import numpy as np
cars = pd.read_csv("Cars.data", delimiter=',', names = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price'])

print(cars.shape)
cars.head()
```

    (205, 26)





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
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>fuel-type</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-wheels</th>
      <th>engine-location</th>
      <th>wheel-base</th>
      <th>...</th>
      <th>engine-size</th>
      <th>fuel-system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>...</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



### Data Cleaning


```python
counter=0
for x in cars['symboling']:
    if x=='?':
        counter=counter+1
print(counter)
counter=0
for x in cars['normalized-losses']:
    if x=='?':
        counter=counter+1
print(counter)
#cars['normalized-losses'].value_counts()['?']
```

    0
    41


#### Number of columns which have missing values in the cars data


```python
counter=0
for x in cars:
    for c in cars[x]:
        if c == '?':
            counter=counter+1
            break
print(counter)
```

    7


#### Total number of missing values in the cars data


```python
counter=0
for x in cars:
    for c in cars[x]:
        if c == '?':
            counter=counter+1
print(counter)
```

    59


#### Replace the missing values using the following strategy:

- For numerical column, replace the missing values with the mean value of that column.
- For categorical column, replace the missing values with the most frequent value of that column.


```python
from sklearn.impute import SimpleImputer
caticorigal_data =['symboling', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system']
notCat = [e for e in cars.columns.values if e not in caticorigal_data]

cars = cars.replace('?', np.nan)

clean_cars = cars.copy()

for x in notCat:
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    clean_cars[x] = imp_mean.fit_transform(clean_cars[[x]]).ravel()

for x in caticorigal_data:
    imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    clean_cars[x] = imp_cat.fit_transform(clean_cars[[x]]).ravel()

clean_cars.head(10)


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
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>fuel-type</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-wheels</th>
      <th>engine-location</th>
      <th>wheel-base</th>
      <th>...</th>
      <th>engine-size</th>
      <th>fuel-system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>122.0</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130.0</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111.0</td>
      <td>5000.0</td>
      <td>21.0</td>
      <td>27.0</td>
      <td>13495.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>122.0</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130.0</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111.0</td>
      <td>5000.0</td>
      <td>21.0</td>
      <td>27.0</td>
      <td>16500.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>122.0</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>...</td>
      <td>152.0</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154.0</td>
      <td>5000.0</td>
      <td>19.0</td>
      <td>26.0</td>
      <td>16500.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>164.0</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>109.0</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102.0</td>
      <td>5500.0</td>
      <td>24.0</td>
      <td>30.0</td>
      <td>13950.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>164.0</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>136.0</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115.0</td>
      <td>5500.0</td>
      <td>18.0</td>
      <td>22.0</td>
      <td>17450.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>122.0</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>136.0</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.5</td>
      <td>110.0</td>
      <td>5500.0</td>
      <td>19.0</td>
      <td>25.0</td>
      <td>15250.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>158.0</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>105.8</td>
      <td>...</td>
      <td>136.0</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.5</td>
      <td>110.0</td>
      <td>5500.0</td>
      <td>19.0</td>
      <td>25.0</td>
      <td>17710.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>122.0</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>wagon</td>
      <td>fwd</td>
      <td>front</td>
      <td>105.8</td>
      <td>...</td>
      <td>136.0</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.5</td>
      <td>110.0</td>
      <td>5500.0</td>
      <td>19.0</td>
      <td>25.0</td>
      <td>18920.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>158.0</td>
      <td>audi</td>
      <td>gas</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>105.8</td>
      <td>...</td>
      <td>131.0</td>
      <td>mpfi</td>
      <td>3.13</td>
      <td>3.40</td>
      <td>8.3</td>
      <td>140.0</td>
      <td>5500.0</td>
      <td>17.0</td>
      <td>20.0</td>
      <td>23875.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>122.0</td>
      <td>audi</td>
      <td>gas</td>
      <td>turbo</td>
      <td>two</td>
      <td>hatchback</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.5</td>
      <td>...</td>
      <td>131.0</td>
      <td>mpfi</td>
      <td>3.13</td>
      <td>3.40</td>
      <td>7.0</td>
      <td>160.0</td>
      <td>5500.0</td>
      <td>16.0</td>
      <td>22.0</td>
      <td>13207.129353</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 26 columns</p>
</div>



#### A histogram on `Price` and findings


```python
import matplotlib.pyplot as plt
plt.hist(clean_cars['price'], bins=5)
plt.title("Prices")
plt.xlabel("Price")
plt.ylabel("#Count")
```

![png](output_15_1.png)

There is a considerably larger cluser of cars in the lower price ranges over the cars in in the larger price ranges.

#### Relationship between `price` and `horsepower`

```python
plt.scatter(clean_cars['price'], clean_cars['horsepower'])
plt.title("Price Horsepower comparision")
plt.xlabel("Price")
plt.ylabel("Horsepower")

#produces line of best fit
plt.plot(np.unique(clean_cars['price']), np.poly1d(np.polyfit(clean_cars['price'], clean_cars['horsepower'], 1))(np.unique(clean_cars['price'])), c='r')
```

![png](output_18_1.png)


Besides from a few outliers there seems to be a strong correlation between the cost of a car and the hourse power provided in the car with a linear increase as hourse power increases so does that of the cost

#### Relationship between `price` and `make`


```python
import seaborn as sns

plt.figure(figsize=(25,5))

sns.set()

gra = sns.violinplot(clean_cars['make'], clean_cars['price'], scale='count', width=1.2)
gra.tick_params(labelsize=20)
gra.xaxis.set_tick_params(rotation=90)
gra.set_xlabel("Make", fontsize=20)
gra.set_ylabel("Price", fontsize=20)
```

![png](output_21_1.png)


Companies that offer primarily higher cost cars tend to offer cars of lower value whereas companies that offer primarily lower cost cars tend to stick to their lower economical price range.

#### Relationship between `price` and `make`


```python
plt.figure(figsize=(10,5))

sns.set()

accepted_Rows = clean_cars['make'].isin(['honda', 'mazda', 'bmw', 'audi', 'toyota'])

gra = sns.violinplot(clean_cars[accepted_Rows]['make'], clean_cars[accepted_Rows]['price'])
gra.set_xlabel("Make", fontsize=14)
gra.set_ylabel("Price", fontsize=14)
```

![png](output_24_1.png)


bmw seems to heavily out preform the other companies shown in terms of the range of prices available.

## KNN Modeling

### Cleaning data


```python
numeric_cars = clean_cars[notCat]
numeric_cars.head(5)
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
      <th>normalized-losses</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb-weight</th>
      <th>engine-size</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>122.0</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>48.8</td>
      <td>2548.0</td>
      <td>130.0</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111.0</td>
      <td>5000.0</td>
      <td>21.0</td>
      <td>27.0</td>
      <td>13495.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>122.0</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>48.8</td>
      <td>2548.0</td>
      <td>130.0</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111.0</td>
      <td>5000.0</td>
      <td>21.0</td>
      <td>27.0</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>122.0</td>
      <td>94.5</td>
      <td>171.2</td>
      <td>65.5</td>
      <td>52.4</td>
      <td>2823.0</td>
      <td>152.0</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154.0</td>
      <td>5000.0</td>
      <td>19.0</td>
      <td>26.0</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>164.0</td>
      <td>99.8</td>
      <td>176.6</td>
      <td>66.2</td>
      <td>54.3</td>
      <td>2337.0</td>
      <td>109.0</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102.0</td>
      <td>5500.0</td>
      <td>24.0</td>
      <td>30.0</td>
      <td>13950.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>164.0</td>
      <td>99.4</td>
      <td>176.6</td>
      <td>66.4</td>
      <td>54.3</td>
      <td>2824.0</td>
      <td>136.0</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115.0</td>
      <td>5500.0</td>
      <td>18.0</td>
      <td>22.0</td>
      <td>17450.0</td>
    </tr>
  </tbody>
</table>
</div>



### normalization of data


```python
from sklearn.preprocessing import Binarizer
pd.options.mode.chained_assignment = None#remove warning because we know it is safe
binVal = numeric_cars['price'].mean()
binarizer = Binarizer(binVal)
numeric_cars['price'] = binarizer.fit_transform(numeric_cars[['price']]).ravel()
pd.options.mode.chained_assignment ='warn'#add warnings back

numeric_cars.head(5)
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
      <th>normalized-losses</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb-weight</th>
      <th>engine-size</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>122.0</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>48.8</td>
      <td>2548.0</td>
      <td>130.0</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111.0</td>
      <td>5000.0</td>
      <td>21.0</td>
      <td>27.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>122.0</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>48.8</td>
      <td>2548.0</td>
      <td>130.0</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111.0</td>
      <td>5000.0</td>
      <td>21.0</td>
      <td>27.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>122.0</td>
      <td>94.5</td>
      <td>171.2</td>
      <td>65.5</td>
      <td>52.4</td>
      <td>2823.0</td>
      <td>152.0</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154.0</td>
      <td>5000.0</td>
      <td>19.0</td>
      <td>26.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>164.0</td>
      <td>99.8</td>
      <td>176.6</td>
      <td>66.2</td>
      <td>54.3</td>
      <td>2337.0</td>
      <td>109.0</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102.0</td>
      <td>5500.0</td>
      <td>24.0</td>
      <td>30.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>164.0</td>
      <td>99.4</td>
      <td>176.6</td>
      <td>66.4</td>
      <td>54.3</td>
      <td>2824.0</td>
      <td>136.0</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115.0</td>
      <td>5500.0</td>
      <td>18.0</td>
      <td>22.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



### Train the KNN model with classification


```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
training_set, test_set = train_test_split(numeric_cars, test_size = 0.25, random_state = 1)

model = KNeighborsClassifier()

X_train = training_set.drop(['price'], axis=1)
y_train = training_set['price']
X_test = test_set.drop(['price'], axis=1) #axis=1 means drop columns
y_test = test_set['price']

model.fit(X_train, y_train)

y_prediction = model.predict(X_test)

from sklearn.metrics import roc_auc_score
print('\033[1m' + "AUC: "+ '\033[0m' + str(roc_auc_score(y_test,y_prediction)))
```

***AUC:*** 0.9126050420168067


### Train the KNN classfication model via different k values.   


```python
kVal = []
res = []
for x in range(1,30):
    model = KNeighborsClassifier(n_neighbors=x)

    model.fit(X_train, y_train)

    y_prediction = model.predict(X_test)
    kVal.append(x)
    res.append(roc_auc_score(y_test,y_prediction))
    
plt.plot(kVal, res)
plt.title("KNN results")
plt.xlabel("K value")
plt.ylabel("Accuracy")
```

![png](output_34_1.png)


### Optimise the parameter k via cross-validated   


```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
parameters = {'n_neighbors':np.arange(1,30)}
clf = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parameters, cv=5, iid=True)
clfRes = clf.fit(X_train, y_train)
#print("==================")
#print(clfRes.best_score_)
#print(clfRes.best_estimator_)
#print(clfRes.best_params_)
#print("==================\n")

print('\033[1m', end ='')
print("The best k value is: ", end = '')
print('\033[0m', end ='')
print(str(clfRes.best_params_['n_neighbors']))

model = KNeighborsClassifier(n_neighbors=clfRes.best_params_['n_neighbors'])

X_train = training_set.drop(['price'], axis=1)
y_train = training_set['price']
X_test = test_set.drop(['price'], axis=1) #axis=1 means drop columns
y_test = test_set['price']

model.fit(X_train, y_train)

y_prediction = model.predict(X_test)

print('\033[1m', end ='')
print("Corresponding AUC performance:")
print('\033[0m', end ='')
print(str(roc_auc_score(y_test,y_prediction)))
```
***The best k value is:*** 7
***Corresponding AUC performance:*** 0.9126050420168067


 ### Train the KNN classification model via 5-fold CV


```python
from sklearn.model_selection import cross_val_score
kVal = []
res = []
all_res = []
for x in range(1,30):
    model = KNeighborsClassifier(n_neighbors=x)
    cv_scores = cross_val_score(model, numeric_cars.drop(['price'], axis=1), numeric_cars['price'], cv=5)
    kVal.append(x)
    res.append(np.mean(cv_scores))
    all_res.append(cv_scores)
plt.plot(kVal, all_res)
plt.title("KNN results")
plt.xlabel("K value")
plt.ylabel("Accuracy")
```

![png](output_38_1.png)

```python
plt.plot(kVal, res)
plt.title("KNN results")
plt.xlabel("K value")
plt.ylabel("Accuracy")
```

![png](output_39_1.png)
