import numpy as np 
import pandas as pd

import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import os
import warnings 
warnings.filterwarnings('ignore')
df = pd.read_csv('googleplaystore.csv')
df.head()
App	Category	Rating	Reviews	Size	Installs	Type	Price	Content Rating	Genres	Last Updated	Current Ver	Android Ver
0	Photo Editor & Candy Camera & Grid & ScrapBook	ART_AND_DESIGN	4.1	159	19M	10,000+	Free	0	Everyone	Art & Design	January 7, 2018	1.0.0	4.0.3 and up
1	Coloring book moana	ART_AND_DESIGN	3.9	967	14M	500,000+	Free	0	Everyone	Art & Design;Pretend Play	January 15, 2018	2.0.0	4.0.3 and up
2	U Launcher Lite – FREE Live Cool Themes, Hide ...	ART_AND_DESIGN	4.7	87510	8.7M	5,000,000+	Free	0	Everyone	Art & Design	August 1, 2018	1.2.4	4.0.3 and up
3	Sketch - Draw & Paint	ART_AND_DESIGN	4.5	215644	25M	50,000,000+	Free	0	Teen	Art & Design	June 8, 2018	Varies with device	4.2 and up
4	Pixel Draw - Number Art Coloring Book	ART_AND_DESIGN	4.3	967	2.8M	100,000+	Free	0	Everyone	Art & Design;Creativity	June 20, 2018	1.1	4.4 and up
print(f'Number of rows : {df.shape [0]}')
print(f'Number of columns : { df.shape[1]}')
Number of rows : 10841
Number of columns : 13
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10841 entries, 0 to 10840
Data columns (total 13 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   App             10841 non-null  object 
 1   Category        10841 non-null  object 
 2   Rating          9367 non-null   float64
 3   Reviews         10841 non-null  object 
 4   Size            10841 non-null  object 
 5   Installs        10841 non-null  object 
 6   Type            10840 non-null  object 
 7   Price           10841 non-null  object 
 8   Content Rating  10840 non-null  object 
 9   Genres          10841 non-null  object 
 10  Last Updated    10841 non-null  object 
 11  Current Ver     10833 non-null  object 
 12  Android Ver     10838 non-null  object 
dtypes: float64(1), object(12)
memory usage: 1.1+ MB
df.duplicated().sum()

print(f"DataFrame has {df.duplicated().sum()} duplicate values")
DataFrame has 483 duplicate values
df.drop_duplicates(inplace=True)

print(f" Total duplicate values : {df.duplicated().sum()}")
 Total duplicate values : 0
df.isnull().sum()
App                  0
Category             0
Rating            1465
Reviews              0
Size                 0
Installs             0
Type                 1
Price                0
Content Rating       1
Genres               0
Last Updated         0
Current Ver          8
Android Ver          3
dtype: int64
## Drop records with nulls in any of the columns.

df.dropna(inplace=True)
df.isnull().sum()
App               0
Category          0
Rating            0
Reviews           0
Size              0
Installs          0
Type              0
Price             0
Content Rating    0
Genres            0
Last Updated      0
Current Ver       0
Android Ver       0
dtype: int64
df.shape
(8886, 13)
# # Variables seem to have incorrect type and inconsistent formatting. You need to fix them:
# Size column has sizes in Kb as well as Mb.
# To analyze, you’ll need to convert these to numeric.
# Extract the numeric value from the column and Multiply the value by 1,000, if size is mentioned in Mb.


def size_col_processing(x):
    x= str(x.lower())
    if 'm' in x:
    
        val=float(x.replace('m', ''))
        val=val*1000
        
    elif 'k'in x:
        val=float(x.replace('k',''))
        
    else:
        val=0
    return val
df['Size']=df['Size'].apply(size_col_processing)
df.head()
App	Category	Rating	Reviews	Size	Installs	Type	Price	Content Rating	Genres	Last Updated	Current Ver	Android Ver
0	Photo Editor & Candy Camera & Grid & ScrapBook	ART_AND_DESIGN	4.1	159	19000.0	10,000+	Free	0	Everyone	Art & Design	January 7, 2018	1.0.0	4.0.3 and up
1	Coloring book moana	ART_AND_DESIGN	3.9	967	14000.0	500,000+	Free	0	Everyone	Art & Design;Pretend Play	January 15, 2018	2.0.0	4.0.3 and up
2	U Launcher Lite – FREE Live Cool Themes, Hide ...	ART_AND_DESIGN	4.7	87510	8700.0	5,000,000+	Free	0	Everyone	Art & Design	August 1, 2018	1.2.4	4.0.3 and up
3	Sketch - Draw & Paint	ART_AND_DESIGN	4.5	215644	25000.0	50,000,000+	Free	0	Teen	Art & Design	June 8, 2018	Varies with device	4.2 and up
4	Pixel Draw - Number Art Coloring Book	ART_AND_DESIGN	4.3	967	2800.0	100,000+	Free	0	Everyone	Art & Design;Creativity	June 20, 2018	1.1	4.4 and up
df['Price']= df['Price'].apply(lambda x :str(x).replace('$','')if '$'in str(x)else str(x))
df['Price']= df['Price'].apply(lambda x : float(x))
df['Reviews']=pd.to_numeric(df['Reviews'], errors ='coerce')  
df['Installs']=df['Installs'].apply(lambda x : str(x).replace('+','') if '+' in str(x) else str(x) )
df['Installs']=df['Installs'].apply(lambda x : str(x).replace(',','') if ',' in str(x) else str(x) )
df['Installs']=df['Installs'].apply(lambda x : float(x))
df.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 8886 entries, 0 to 10840
Data columns (total 13 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   App             8886 non-null   object 
 1   Category        8886 non-null   object 
 2   Rating          8886 non-null   float64
 3   Reviews         8886 non-null   int64  
 4   Size            8886 non-null   float64
 5   Installs        8886 non-null   float64
 6   Type            8886 non-null   object 
 7   Price           8886 non-null   float64
 8   Content Rating  8886 non-null   object 
 9   Genres          8886 non-null   object 
 10  Last Updated    8886 non-null   object 
 11  Current Ver     8886 non-null   object 
 12  Android Ver     8886 non-null   object 
dtypes: float64(4), int64(1), object(8)
memory usage: 971.9+ KB
df.describe()
Rating	Reviews	Size	Installs	Price
count	8886.000000	8.886000e+03	8886.000000	8.886000e+03	8886.000000
mean	4.187959	4.730928e+05	19000.655919	1.650061e+07	0.963526
std	0.522428	2.906007e+06	23023.418686	8.640413e+07	16.194792
min	1.000000	1.000000e+00	0.000000	1.000000e+00	0.000000
25%	4.000000	1.640000e+02	2500.000000	1.000000e+04	0.000000
50%	4.300000	4.723000e+03	9400.000000	5.000000e+05	0.000000
75%	4.500000	7.131325e+04	27000.000000	5.000000e+06	0.000000
max	5.000000	7.815831e+07	100000.000000	1.000000e+09	400.000000
# Reviews should not be more than installs as only those who installed can review the app.
# If there are any such records, drop them.

df['review_check']=df['Reviews']>df['Installs']
df.shape
(8886, 14)
df[df['review_check']==True].head(2)
App	Category	Rating	Reviews	Size	Installs	Type	Price	Content Rating	Genres	Last Updated	Current Ver	Android Ver	review_check
2454	KBA-EZ Health Guide	MEDICAL	5.0	4	25000.0	1.0	Free	0.00	Everyone	Medical	August 2, 2018	1.0.72	4.0.3 and up	True
4663	Alarmy (Sleep If U Can) - Pro	LIFESTYLE	4.8	10249	0.0	10000.0	Paid	2.49	Everyone	Lifestyle	July 30, 2018	Varies with device	Varies with device	True
df=df[df['review_check']== False]
df.shape
(8879, 14)
df['review_check'].unique()
array([False])
df.drop('review_check',axis=1,inplace=True)
df.head(1)
App	Category	Rating	Reviews	Size	Installs	Type	Price	Content Rating	Genres	Last Updated	Current Ver	Android Ver
0	Photo Editor & Candy Camera & Grid & ScrapBook	ART_AND_DESIGN	4.1	159	19000.0	10000.0	Free	0.0	Everyone	Art & Design	January 7, 2018	1.0.0	4.0.3 and up
# For free apps (type = “Free”), the price should not be >0. Drop any such rows.

df[(df['Type']=='Free')&(df['Price']>0)]
App	Category	Rating	Reviews	Size	Installs	Type	Price	Content Rating	Genres	Last Updated	Current Ver	Android Ver
# ~ is used to Negate/reverse the df selected using condition 

df=df[~((df['Type']=='Free')& (df['Price']>0))]
df.shape
(6981, 13)
plt.figure(figsize=(12,6))
df.boxplot(column ='Price')
<AxesSubplot:>

# outlier are present in dataset , anything above than 300 will be concider as outlier 
plt.figure(figsize=(12,6))
df.boxplot('Reviews')
<AxesSubplot:>

# values above than 3 to le7 are the outliers in box plot shown above 
plt.figure(figsize=(12,6))
df.hist('Rating')
array([[<AxesSubplot:title={'center':'Rating'}>]], dtype=object)
<Figure size 864x432 with 0 Axes>

# Histogram for Size

sns.histplot(df['Size'])
<AxesSubplot:xlabel='Size', ylabel='Count'>

# Most(50%) of the apps are below 20MB of size.
# From the box plot, it seems like there are some apps with very high price.
# A price of $200 for an application on the Play Store is very high and suspicious!
# Check out the records with very high price.
# Is 200 indeed a high price?

df=df[df['Price']>200]
df
App	Category	Rating	Reviews	Size	Installs	Type	Price	Content Rating	Genres
4197	most expensive app (H)	FAMILY	4.3	6	1500.0	100.0	Paid	399.99	Everyone	Entertainment
4362	💎 I'm rich	LIFESTYLE	3.8	718	26000.0	10000.0	Paid	399.99	Everyone	Lifestyle
4367	I'm Rich - Trump Edition	LIFESTYLE	3.6	275	7300.0	10000.0	Paid	400.00	Everyone	Lifestyle
5351	I am rich	LIFESTYLE	3.8	3547	1800.0	100000.0	Paid	399.99	Everyone	Lifestyle
5354	I am Rich Plus	FAMILY	4.0	856	8700.0	10000.0	Paid	399.99	Everyone	Entertainment
5355	I am rich VIP	LIFESTYLE	3.8	411	2600.0	10000.0	Paid	299.99	Everyone	Lifestyle
5356	I Am Rich Premium	FINANCE	4.1	1867	4700.0	50000.0	Paid	399.99	Everyone	Finance
5357	I am extremely Rich	LIFESTYLE	2.9	41	2900.0	1000.0	Paid	379.99	Everyone	Lifestyle
5358	I am Rich!	FINANCE	3.8	93	22000.0	1000.0	Paid	399.99	Everyone	Finance
5359	I am rich(premium)	FINANCE	3.5	472	965.0	5000.0	Paid	399.99	Everyone	Finance
5362	I Am Rich Pro	FAMILY	4.4	201	2700.0	5000.0	Paid	399.99	Everyone	Entertainment
5364	I am rich (Most expensive app)	FINANCE	4.1	129	2700.0	1000.0	Paid	399.99	Teen	Finance
5366	I Am Rich	FAMILY	3.6	217	4900.0	10000.0	Paid	389.99	Everyone	Entertainment
5369	I am Rich	FINANCE	4.3	180	3800.0	5000.0	Paid	399.99	Everyone	Finance
5373	I AM RICH PRO PLUS	FINANCE	4.0	36	41000.0	1000.0	Paid	399.99	Everyone	Finance
9917	Eu Sou Rico	FINANCE	4.4	0	1400.0	0.0	Paid	394.99	Everyone	Finance
9934	I'm Rich/Eu sou Rico/أنا غني/我很有錢	LIFESTYLE	4.4	0	40000.0	0.0	Paid	399.99	Everyone	Lifestyle
# Yes $200 indeed  is a high price.
# Drop these as most seem to be junk apps

df=df[df['Price']<200]
df['Price'].unique()
array([ 0.  ,  4.99,  3.99,  6.99,  7.99,  5.99,  2.99,  3.49,  1.99,
        9.99,  7.49,  0.99,  9.  ,  5.49, 10.  , 24.99, 11.99, 79.99,
       16.99, 14.99, 29.99, 12.99,  2.49, 10.99,  1.5 , 19.99, 15.99,
       33.99, 39.99,  3.95,  4.49,  1.7 ,  8.99,  1.49,  3.88, 17.99,
        3.02,  1.76,  4.84,  4.77,  1.61,  2.5 ,  1.59,  6.49,  1.29,
       37.99, 18.99,  8.49,  1.75, 14.  ,  2.  ,  3.08,  2.59, 19.4 ,
        3.9 ,  4.59, 15.46,  3.04, 13.99,  4.29,  3.28,  4.6 ,  1.  ,
        2.95,  2.9 ,  1.97,  2.56,  1.2 ])
# Reviews: Very few apps have very high number of reviews. These are all star apps that don’t help with the analysis and, 
# in fact, will skew it. Drop records having more than 2 million reviews.

df=df[df['Reviews']<2000000]
df.head(2)
App	Category	Rating	Reviews	Size	Installs	Type	Price	Content Rating	Genres	Last Updated	Current Ver	Android Ver
0	Photo Editor & Candy Camera & Grid & ScrapBook	ART_AND_DESIGN	4.1	159	19000.0	10000.0	Free	0.0	Everyone	Art & Design	January 7, 2018	1.0.0	4.0.3 and up
1	Coloring book moana	ART_AND_DESIGN	3.9	967	14000.0	500000.0	Free	0.0	Everyone	Art & Design;Pretend Play	January 15, 2018	2.0.0	4.0.3 and up
# Installs:  There seems to be some outliers in this field too.
# Find out the different percentiles – 10, 25, 50, 70, 90, 95, 99
# Decide a threshold as cutoff for outlier and drop records having values more than that


df.Installs.quantile([0.10, 0.25, 0.50, 0.70, 0.90, 0.95, 0.99])
0.10         1000.0
0.25        10000.0
0.50       100000.0
0.70      1000000.0
0.90     10000000.0
0.95     10000000.0
0.99    100000000.0
Name: Installs, dtype: float64
# Keeping 95% value as a threshold/cutoff for outlier and drop records having values more than that.

df=df[df['Installs']<10000000.0]
df.head(2)
App	Category	Rating	Reviews	Size	Installs	Type	Price	Content Rating	Genres	Last Updated	Current Ver	Android Ver
0	Photo Editor & Candy Camera & Grid & ScrapBook	ART_AND_DESIGN	4.1	159	19000.0	10000.0	Free	0.0	Everyone	Art & Design	January 7, 2018	1.0.0	4.0.3 and up
1	Coloring book moana	ART_AND_DESIGN	3.9	967	14000.0	500000.0	Free	0.0	Everyone	Art & Design;Pretend Play	January 15, 2018	2.0.0	4.0.3 and up
df.shape
(6981, 13)
# Make scatter plot/joinplot for Rating vs. Price
# What pattern do you observe? Does rating increase with price?
# Yes, it is showing positive correlation as the price increasing Ratings also increase.

# Make scatter plot/joinplot for Rating vs. Size
# Are heavier apps rated better?
# No relation as we can see everyone is downloading any size of the app.

# Make scatter plot/joinplot for Rating vs. Reviews
# Does more review mean a better rating always?
# Apps which are having higher ratings
# The app which are having higher rating are getting somewhat of a more reviews. 
# Most of the ratings are on the higher end side of the ratings.
plt.figure(figsize=(12,6))
df.boxplot('Installs') 
<AxesSubplot:>

sns.pairplot(df)
<seaborn.axisgrid.PairGrid at 0x1832d8b8310>

##  value count of top most app on google 
 
x = df['Category'].value_counts()
y = df['Category'].value_counts().index 

x_axis =[]
y_axis = []
for i in range(len(x)):
    x_axis.append(x[i])
    y_axis.append(y[i])
plt.figure(figsize=(18,13))
plt.xlabel("Count")
plt.ylabel("Category")

graph = sns.barplot(x = x_axis, y = y_axis, palette= "husl")
graph.set_title("Top categories on Google Playstore", fontsize = 25);

## Bar chart for Content Rating 

x1 = df['Content Rating'].value_counts().index
y1 = df['Content Rating'].value_counts()

x1_axis = []
y1_axis = []

for i in range(len(x1)):
    x1_axis.append(x1[i])
    y1_axis.append(y1[i])
plt.figure(figsize=(12,10))
sns.barplot(x= x1_axis, y= y1_axis)
plt.title('Content Rating',size = 20);
plt.ylabel('Apps(Count)');
plt.xlabel('Content Rating');

## Scatter plot for rating Vs Price
plt.figure(figsize=(12,6))
sns.scatterplot(x='Rating',y='Price',data=df)
<AxesSubplot:xlabel='Rating', ylabel='Price'>

## Scatter plot for rating vs Reviews
plt.figure(figsize=(12,6))
sns.scatterplot(x='Rating',y='Reviews',data=df)
<AxesSubplot:xlabel='Rating', ylabel='Reviews'>

plt.figure(figsize=[12,5])
sns.boxplot("Rating", "Content Rating", data=df )
<AxesSubplot:xlabel='Rating', ylabel='Content Rating'>

plt.figure(figsize=[12,14])
sns.boxplot("Rating", "Category", data=df )
<AxesSubplot:xlabel='Rating', ylabel='Category'>

# Make boxplot for Ratings vs. Category
# Which genre has the best ratings?
# Here Q2 (Median) is higher in 'BOOKS_AND_REFERENCES' and 'EVENTS' and has best with 4.5 ratings.
# For the steps below, create a copy of the dataframe to make all the edits. Name it inp1.
inp1 =df.copy()
inp1
App	Category	Rating	Reviews	Size	Installs	Type	Price	Content Rating	Genres	Last Updated	Current Ver	Android Ver
0	Photo Editor & Candy Camera & Grid & ScrapBook	ART_AND_DESIGN	4.1	159	19000.0	10000.0	Free	0.0	Everyone	Art & Design	January 7, 2018	1.0.0	4.0.3 and up
1	Coloring book moana	ART_AND_DESIGN	3.9	967	14000.0	500000.0	Free	0.0	Everyone	Art & Design;Pretend Play	January 15, 2018	2.0.0	4.0.3 and up
2	U Launcher Lite – FREE Live Cool Themes, Hide ...	ART_AND_DESIGN	4.7	87510	8700.0	5000000.0	Free	0.0	Everyone	Art & Design	August 1, 2018	1.2.4	4.0.3 and up
4	Pixel Draw - Number Art Coloring Book	ART_AND_DESIGN	4.3	967	2800.0	100000.0	Free	0.0	Everyone	Art & Design;Creativity	June 20, 2018	1.1	4.4 and up
5	Paper flowers instructions	ART_AND_DESIGN	4.4	167	5600.0	50000.0	Free	0.0	Everyone	Art & Design	March 26, 2017	1.0	2.3 and up
...	...	...	...	...	...	...	...	...	...	...	...	...	...
10833	Chemin (fr)	BOOKS_AND_REFERENCE	4.8	44	619.0	1000.0	Free	0.0	Everyone	Books & Reference	March 23, 2014	0.8	2.2 and up
10834	FR Calculator	FAMILY	4.0	7	2600.0	500.0	Free	0.0	Everyone	Education	June 18, 2017	1.0.0	4.1 and up
10836	Sya9a Maroc - FR	FAMILY	4.5	38	53000.0	5000.0	Free	0.0	Everyone	Education	July 25, 2017	1.48	4.1 and up
10837	Fr. Mike Schmitz Audio Teachings	FAMILY	5.0	4	3600.0	100.0	Free	0.0	Everyone	Education	July 6, 2018	1.0	4.1 and up
10839	The SCP Foundation DB fr nn5n	BOOKS_AND_REFERENCE	4.5	114	0.0	1000.0	Free	0.0	Mature 17+	Books & Reference	January 19, 2015	Varies with device	Varies with device
6981 rows × 13 columns

df['Reviews'].describe()
count      6981.000000
mean      18564.907606
std       47341.662556
min           1.000000
25%          78.000000
50%        1213.000000
75%       15192.000000
max      896118.000000
Name: Reviews, dtype: float64
## apply log transformation (np.Log1p) on Reviwes and Installs 
inp1['Reviews']=np.log1p(inp1['Reviews'])
inp1['Installs']=np.log1p(inp1['Installs'])
inp1.head(1)
App	Category	Rating	Reviews	Size	Installs	Type	Price	Content Rating	Genres	Last Updated	Current Ver	Android Ver
0	Photo Editor & Candy Camera & Grid & ScrapBook	ART_AND_DESIGN	4.1	5.075174	19000.0	9.21044	Free	0.0	Everyone	Art & Design	January 7, 2018	1.0.0	4.0.3 and up
## droping the columns which are not usefull for further working
inp1.drop(['App','Last Updated','Current Ver','Android Ver'], axis=1,inplace=True)
inp1.head(2)
Category	Rating	Reviews	Size	Installs	Type	Price	Content Rating	Genres
0	ART_AND_DESIGN	4.1	5.075174	19000.0	9.210440	Free	0.0	Everyone	Art & Design
1	ART_AND_DESIGN	3.9	6.875232	14000.0	13.122365	Free	0.0	Everyone	Art & Design;Pretend Play
# Get dummy columns for Category, Genres, and Content Rating. 

inp2=pd.get_dummies(inp1,columns=['Content Rating','Genres','Category','Type'])
inp2.head(2)
Rating	Reviews	Size	Installs	Price	Content Rating_Adults only 18+	Content Rating_Everyone	Content Rating_Everyone 10+	Content Rating_Mature 17+	Content Rating_Teen	...	Category_PRODUCTIVITY	Category_SHOPPING	Category_SOCIAL	Category_SPORTS	Category_TOOLS	Category_TRAVEL_AND_LOCAL	Category_VIDEO_PLAYERS	Category_WEATHER	Type_Free	Type_Paid
0	4.1	5.075174	19000.0	9.210440	0.0	0	1	0	0	0	...	0	0	0	0	0	0	0	0	1	0
1	3.9	6.875232	14000.0	13.122365	0.0	0	1	0	0	0	...	0	0	0	0	0	0	0	0	1	0
2 rows × 156 columns

x=inp2.drop('Rating',axis=1)   # independent Variable
y=inp2['Rating']                # Dependent Variable
from sklearn.model_selection import train_test_split
# Train test split  and apply 70-30 split. Name the new dataframes df_train and df_test.
# Separate the dataframes into X_train, y_train, X_test, and y_test.


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.30, random_state=42)


from sklearn.linear_model import LinearRegression as LR
x_train.head(1)
Reviews	Size	Installs	Price	Content Rating_Adults only 18+	Content Rating_Everyone	Content Rating_Everyone 10+	Content Rating_Mature 17+	Content Rating_Teen	Content Rating_Unrated	...	Category_PRODUCTIVITY	Category_SHOPPING	Category_SOCIAL	Category_SPORTS	Category_TOOLS	Category_TRAVEL_AND_LOCAL	Category_VIDEO_PLAYERS	Category_WEATHER	Type_Free	Type_Paid
9588	7.955425	39000.0	13.122365	0.0	0	1	0	0	0	0	...	0	0	0	0	0	0	0	0	1	0
1 rows × 155 columns

# Use linear regression as the technique

model=LR()
model.fit(x_train, y_train)
LinearRegression()
# Report the R2 on the train set

model.score(x_train, y_train)
0.15268919030909045
# Make predictions on test set and report R2.

model.score(x_test, y_test)
0.11400450481740809
model.predict(x_test)
array([3.74056292, 4.03368424, 4.14940299, ..., 4.21011289, 4.29769173,
       4.27070364])
 
