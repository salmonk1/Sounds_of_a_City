#Noise Complaints in NYC: Which neighborhood is the noisiest (or whiniest?)

Table of Contents
Introduction
Data Sources
Data Wrangling/EDA
Feature Engineering
Modeling
Model Evaluation
Conclusion
Limitations
Next Steps

Libraries
Pandas
NumPy
BeautifulSoup
Re
SciPy
scikit-learn
Matplotlib
seaborn

Introduction

In New York, there are two kinds of noise: the sounds of the city (car horns, loud neighbors, construction equipment, barking dogs) and the sound of New Yorkers complaining about it. In 2007, the city modernized its noise code for the first time in thirty years in order to measure the effect of noise on residents’ quality of life. 

Today, in order to file a complaint, New Yorkers simply have to dial 311 and they are making great use of this hotline. In 2014, the city logged over 140,000 complaints and in 2020 the city logged over 800,000 - more than one complaint per minute.

Noise complaints are especially interesting because they can reflect differences in class, race and culture and provide insights into how a neighborhood is changing as different demographics of people move around the city and live within close proximity to each other.

Noise affects everyone, no matter who you are or where you live and what is one person’s noise is simply the sound of another person’s daily life.

Hypothesis
A neighborhood with the most noise complaints isn’t the noisiest neighborhood; it is the neighborhood with the most self-selection bias. My hypothesis is that neighborhoods with the most noise complaints are wealthier on average and that I can predict whether or not property values in a neighborhood are above or below average based on the number and types of noise complaints logged in that neighborhood.

Data Sources: Noise Complaints and Property Sales from 2016-2020

Open Data NYC 
At the end of every day, all complaints logged with 311 are automatically uploaded to the online dataset where accessing them through the API is a breeze. 
Each observation includes the type of complaint and descriptor of the noise, agency assigned to resolve complaint, resolution status and location information
Data available from 2007

NYC Department of Finance
Annualized Sales files are available for yearly sales information of properties sold in New York City. These files also have information such as neighborhood, building type, zoning class, and other data.
The files are downloadable as one Excel spreadsheet per borough per year
Data available from 2003

Using data from 2016-2020 ensured that I would have a workable amount of observations. I decided I would make a new category to merge my two datasets which would be a combination of post code and year eg. 11221-2016, 11221-2017, 11221-2018, etc.







Data Wrangling and Exploratory Data Analysis

The noise complaint data required little cleaning and was ready for EDA. After dropping unnecessary columns, I had 2,199,949 observations (unique complaints) from the last 5 years and 34 features to use to predict sale prices:























Residential Complaints and Loud Music/Party have the highest count and they also appear in the same areas: Upper Manhattan and Central Brooklyn. As you can see from the maps below, noise complaints give a surprisingly detailed picture of the characteristics of a neighborhood.



























Wrangling the property sales data into usable condition required a lot of attention.


I started with one Excel sheet per borough per year. My goal was to narrow all of this down to an average sale price of a residential unit (apartment unit, single family house), per postcode per year. What I refer to as PPU - price per unit.

Each year of sales had to be dealt with separately because the organization of the data varied from year to year. I needed only residential property sales but the data included sales for commercial property, industrial, mixed-use, whole blocks of flats, single unit flats and homes. Using different tax and building class categories, I was able to filter down to only residential homes and flat blocks which gave me a mix of single unit sales and multi-flat sales. By analyzing the characters in the text addresses of these properties with Regex, I was able to differentiate between these different types of sales. In the case of, say, a 150 unit apartment building sold for 24 million dollars, I divided the total building price by the number of units to get an estimate of a unit sale. 

Another challenge was dealing with outliers and abnormal transactions. There were many sales that were clearly not occuring at market rate. Say, $0 for an apartment that changed hands via inheritance or a business selling a building back to itself for $1 or $10. Also, many sales occurred below market rate in buildings that the housing authority has restricted for people of certain income ranges.

On the other end of the spectrum, there were a huge amount of outlier sales that did occur at market rate costing over $15,000,000.

In the end, I removed all the $0 dollar sales and kept the outliers. By the time I took the average price per postcode and then used this as the target for a binary or multi-class classifier (ie. if the price was high/low or if the price fell into quartiles) the effect of the outliers on my scores was negligible.

My unit sales ranged from $1 to $249 million. With a mean of $1.16 million and median of $642,000. 

My range of mean sales per zip code per year went from $138,000 to $5,171,098 with a mean of  $757,462.70 and median of $512,594.



















Feature Engineering

After running a Chi-Square test, I determined that there was a multicollinearity between features, as expected. I condensed similar features into single categories. For example, ‘car/truck music’, ‘car/truck honking’, ‘engine idling’ were condensed into a single category ‘Vehicle Noise’. In addition to running models on all features together, I also split the features into ‘Complaint Types’ and ‘Descriptors’ and ran PCA on them to see if this would improve my scores and it did not. 

I also dummified my features and my target.


Residential Noise and Party/Music Noise show a clearer connection with low price. Loud Talking and Street Noise are closely connected. Vehicle complaint type and descriptions of vehicle-related noise are now perfectly correlated.































Modeling

For my initial modeling, I used binary classifiers based on the mean sale price and then on the median sale price as the targets with the aim of predicting if the price was above or below either of these averages. There was a big class imbalance when I used the mean as the target so I ultimately decided to use the median. Using the median as the target gave me a 24% score increase over baseline even though the mean target technically has the highest score but a smaller increase from baseline.

Later, I changed my target to a multiclass where I attempted to predict quartiles. Because these were computationally heavy, I ran a RandomizedSearchCV to find the best parameters for a Logistic Regression classifier, which yielded terrible scores but the k-NN model showed promising results. 

	

Median baseline with binary target (High price majority) = 0.5047

Mean Baseline with binary target (Low price majority) = 0.7087

Multiclass baseline split along quartiles:
3    0.253165
4    0.253165
1    0.246835
2    0.246835



Model
Best Test Score
Decision Tree Classifier, Median
0.6296
Decision Tree Classifier, Mean
0.7481




Logistic Regression with Median as Target


Logistic Regression, all feats
0.6289
Logistic Regression, consolidated features
0.6201
Logistic Regression, descriptors
0.5783
Logistic Regression, complaint types
0.6184




Logistic Regression with Mean as Target


Logistic Regression, all feats
0.7473
Logistic Regression, consolidated features
0.7456
Logistic Regression, descriptors
0.7433
Logistic Regression, complaint types
0.7231




Multiclass, target split along quartiles


Logistic Regression 
0.03125
k-NN
0.2468



Model Evaluation



The logistic regression classifier gives insight into the impact of each coefficient on the target variable. The coefficients with a negative impact are all construction related and they correspond to the likelihood of the median sale price being high. If you look at the maps above of construction noise complaints, you can see that the most construction noise occurs in the most expensive neighborhoods of Manhattan.








Extracting the feature importances from the Decision Tree Classifier confirms what we already know, which is that Residential Noise has a huge impact on housing prices. This makes sense because this complaint type makes up over 50% of all complaints.

Confusion Matrix and ROC Curve for Decision Tree Classifier

Label 1 = low price (below median)
Label 2 = high price (above median)


This is the breakdown of the model’s correct and incorrect predictions. We can see that the model is correct the majority of the time. 


Conclusion
In the first part of my hypothesis, I predicted that residents of wealthier neighborhoods file the most noise complaints and this is not true based on my findings. However, it would be interesting to investigate whether 
neighborhoods with the fastest rate of change in property values emit greater noise complaints. Another way to phrase the question: Do noise complaints signal the gentrification of a neighborhood?

As for the second half of my thesis, using a Decision Tree Classifier, I can predict with 63% accuracy whether the price of a home will be higher or lower than the median price of neighborhood property values based on the 3noise complaints made in that neighborhood. This is better than randomly guessing (or better than the accuracy score of 50%). 


Limitations
Number of observations and categories make running even very simple models very time consuming
Skewed distribution of sale prices
Some dependency among features 
No streamlined method to deal with outliers (abnormal sales)


Key Learning
Wrangling my raw property sales data down to a single average price per postcode required hours of cleaning. It was a great opportunity for a deep dive into Pandas and NumPy.
Outliers: The range of my property sales data was huge ($1 - $254 million) and it was heavily skewed. This meant that the usual methods for removing outliers (taking 3 standard deviations from the median, multiplying the IQR by 1.5 or cutting the outer 10% of values) either extended my range into negative prices or cut off too much of the lower end of my data. I spent a lot of time experimenting with different methods for eliminating outliers but in the end I found that keeping them had a minimal effect on my models.
I was concerned about multicollinearity even after I consolidated my features. I ran models with different feature selections but in the end I found that my best models used all features.
Tableau is wonderful for mapping visualizations.


Next Steps
Tune k-NN multiclass model and experiment with additional multiclass models
Experiment with additional feature engineering
Run additional models on only the IQR of sale prices in order to deal with outlier challenges
Explore specific clusters of complaints 
Calculate the rate of change of property values and use this as the target.



