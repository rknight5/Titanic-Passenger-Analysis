
### Import neccessary libraries:

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
%matplotlib inline

### Load the dataset 'titanic.csv' into a Pandas DataFrame and call it `df`.

path_to_file = "titanic.csv"
df=pd.read_csv(path_to_file)

### Show the head of `df` and give a high-level overview of the contents of the DataFrame.

df.head()

df.tail()

### Run the `info` method on the DataFrame and explain the meaning of each of the columns.

df.info()
#The summary includes a full list of all the columns, their data types, the number of non-null values in each column.  
#The value of rangeindex is also provided for the index axis.

### Use the `describe` method to obtain the main statistical features of the data.

df.describe()

### Include 50% to 100% percentile with 5% increments for the 'age' column in the `describe` method output

df['age'].describe(percentiles=[.5,.55,.60,.65,.70,.75,.80,.85,.90,.95,1])

### What is the 80% percentile of the 'age' column?

df.age.quantile(0.8)

### Create a new DataFrame, call it `df1`, that only stores passengers whose age above the 80% percentile calculated above.

df1 = df[df.age > df.age.quantile(.80)]
df1

### Back to the original DataFrame `df`. Count the number of passengers in first class.

df['pclass'].value_counts().sort_index()
#323 passengers in first class cabins

### Create a copy of the original DataFrame, call it `df2`, and insert a new column called 'older than 90% percentile' that stores true/false if the passenger is older/younger than the 90% percentile.

df2 = df

old = df.age.quantile(0.9)
print("The 90th percentile age is:",old)
old_age = df.age >= old
df2['older than 90% percentile'] = old_age
df2

### Back to the original DataFrame `df`. Group the 'fare' column by the class of passenger and get the mean fare price for each class

df.groupby("pclass", sort=True)["fare"].mean().round(2)

### Plot the bar chart for the avg fare grouped by class

fare_ave = df.groupby("pclass", sort=True)["fare"].mean().round(2)
fare_ave.plot.bar(x='pclass', y='val', rot=0)

### Getting the number of outliers: define the upper bound and lower bound for the 'fare' column at 95% or 5% and count how many passengers are considered outliers

print(df.fare.quantile(.05))
print(df.fare.quantile(.95))

dfout = df[df.fare.between(df.fare.quantile(.05), df.fare.quantile(.95))]
len(df.axes[0]) - len(dfout.axes[0])
#121 outliers 

### Replace outliers in the dataframe by the mean of the 'fare' column

mean = df['fare'].mean()
df.loc[df.fare >= 133.65, 'fare'] = np.nan
df.loc[df.fare <= 7.225, 'fare'] = np.nan
df.fillna(mean,inplace=True)

df

### Load the contents of 'titanic2.csv' into a new Pandas DataFrame, call it `df3`. Examine the data using the `head` method. Combine the data with the original dataframe `df`. Call this combined version `df4`. We will be working with `df4` and building on it for the remainder of the exam.

path_to_file2 = "titanic2.csv"
df3=pd.read_csv(path_to_file2)
df3.head()

df4 = df.append(df3, ignore_index=True, sort=False)
len(df4.axes[0])

### Plot a scatter plot of 'age' vs 'fare' for your data and calculate the correlation between these two columns

plt.scatter(df4['age'], df4['fare'])

df4['age'].corr(df4['fare']).round(6)

df4[['age','fare']].corr()

### Drop the columns: 'name', 'cabin', 'boat', 'body','home.dest', 'embarked', 'ticket'

len(df4.axes[1])

df4.drop(['name', 'cabin', 'boat', 'body', 'home.dest', 'embarked', 'ticket'], axis=1, inplace=True)

len(df4.axes[1])

df4.head()

# Filling missing data in 'age' column by using an educated guess:

### Write a function that takes two columns, 'age' and 'pclass', as input and depending on which pclass, if the age is `NaN` (command: pd.isnull(age)   ) it returns age 40, 30 or 25. For example, if input is [`NaN`, 1], it returns age = 40

#Answer here
df4.age.isna().value_counts()

pd.isnull(df4['age']).value_counts()

def clean(age, pclass):
    if pd.isna(age):
        if pclass == 1:
            age = 40
        elif pclass == 2:
            age = 30
        elif pclass == 3: 
            age = 25
    return(age)

df4.loc[(pd.isnull(df4["age"])) & (df4["pclass"] == 1), "age"]=40
df4.loc[(pd.isnull(df4["age"])) & (df4["pclass"] == 2), "age"]=30
df4.loc[(pd.isnull(df4["age"])) & (df4["pclass"] == 3), "age"]=25
pd.isnull(df4['age']).value_counts()

### Update the age column in the DataFrame using the `apply` method to clean up the `NaNs` using the function created above

df4['age'] = df4.apply(lambda x: clean(x['age'],x['pclass']), axis=1)

pd.isna(df4['age']).value_counts()

df4

### Display all rows where there is a `NaN` in the fare column.

df4[pd.isna(df4['fare'])]

df4[pd.isnull(df4['fare'])]

### Update the sex column in the DataFrame using `apply` method: replace the value `female` with 1 and `male` with 0.

df4['sex']=df4['sex'].replace(['female','male'], [1,0])

df4.sex.isna().value_counts()

### Display the top four rows of the DataFrame.

df4.head(4)

### Confirm that the contents of the 'sex' column are integers and not strings. (There are several different ways of doing this.)

df4.sex.apply(type)

df4['sex'] = df4['sex'].apply(np.int64)

df4['sex'].apply(type)

### Replace the `NaN` values in the 'fare' column, with the mean of the column. Compute and display the mean of the updated column.

df4['fare'].mean()

df['fare'].fillna(df['fare'].mean())

df4['fare'].mean()

### Drop any rows that have a `NaN`.

pd.isna(df4).sum(axis=0)

df4 = df4.dropna()

pd.isna(df4).sum(axis=0)

### Display the head of cleaned dataframe.

df4.head()

### Display the number of rows and columns of the dataframe

print(len(df4.axes[1]))
print(len(df4.axes[0]))
