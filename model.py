'''
This model recommends books from the dataset using the best algorithm
as evaluated using the lowest RMSE. It also aims to generate book titles
alongside the recommended ISBN numbers.
'''

import pandas as pd
import numpy as np

from src.utilities import *
import surprise
from surprise import SVD
from surprise import Reader
from surprise import Dataset
from surprise import BaselineOnly, SVD, NMF, KNNBasic, KNNBaseline, KNNWithMeans, NormalPredictor
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy
from collections import defaultdict

# Import CSV files
ratings_df = csv('https://markg110.s3-us-west-1.amazonaws.com/data/BX-Book-Ratings.csv')
books_df = csv('https://markg110.s3-us-west-1.amazonaws.com/data/BX-Books.csv')
users_df = csv('https://markg110.s3-us-west-1.amazonaws.com/data/BX-Users.csv')

# Limited data wrangling
books_new = books_df.copy()
books_new.dropna(inplace=True)
columns = ['Book-Author', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
books_df.drop(columns=columns, inplace=True)
books_df.rename(columns={'Year-Of-Publication':'Publication-Year'},inplace=True)
books_ratings_df = pd.merge(ratings_df, books_df, on='ISBN')
print(books_ratings_df.shape)
print('===========================================================')
books_ratings_df.head()

# Removing irrelevant data and casting publication year to int64
books_ratings_df = books_ratings_df[books_ratings_df['Publication-Year'] != 'Gallimard']
books_ratings_df = books_ratings_df[books_ratings_df['Publication-Year'] != 'DK Publishing Inc']
books_ratings_df['Publication-Year'] = books_ratings_df['Publication-Year'].apply(lambda x: int(x))

# Isolating most rated books which were published between 1975 and 2002
books_ratings_df = books_ratings_df[books_ratings_df['Publication-Year'] >= 1975]
books_ratings_df = books_ratings_df[books_ratings_df['Publication-Year'] <= 2002]

min_book_ratings = 50 # Books: At least 50 ratings received
min_user_ratings = 50 # Users: At least 50 ratings given

# Trimming down dataset
filter_books = books_ratings_df['ISBN'].value_counts() > min_book_ratings # Isolating books with more than 50 ratings received
filter_books = filter_books[filter_books].index.tolist() # Isolating ISBNs
filter_users = books_ratings_df['User-ID'].value_counts() > min_user_ratings # Isolating users with more than 50 ratings given
filter_users = filter_users[filter_users].index.tolist() # Isolating User-IDs

# Filtering dataframe based on minimum number of ratings by ISBNs and User-IDs
filtered_df = books_ratings_df[(books_ratings_df['ISBN'].isin(filter_books)) & (books_ratings_df['User-ID'].isin(filter_users))]
columns=['Book-Title', 'Publication-Year']
filtered_df.drop(columns=columns, inplace=True)

'''
=========================================================
** MODELING **
=========================================================
'''

# Instantiate Surprise classes
reader = Reader(rating_scale=(0,10))
data = Dataset.load_from_df(filtered_df, reader)

# Algorithm iterator for cross_validation
algorithm = [BaselineOnly(), SVD(), KNNBasic(), KNNBaseline(), KNNWithMeans(), NormalPredictor()]

comparison_list = []
for algo in algorithm:
    # Perform cross validation with RMSE as evaluation metric
    results = cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=False)
    
    # Append to comparison list
    get_result = pd.DataFrame.from_dict(results).mean(axis=0)
    get_result = get_result.append(pd.Series([str(algo).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    comparison_list.append(get_result)

# Generating cross validation results
# We choose BaselineOnly algorithm as it yields the lowest RMSE
surprise_results = pd.DataFrame(comparison_list).set_index('Algorithm').sort_values('test_rmse')
print('')
print('Here are the results of running cross validation on six algorithms namely, \n
BaselineOnly, SVD, KNNBasic, KNNWithMeans, and NormalPredictor')
print('')
surprise_results

# Configuring baselines using alternating least squares (ALS) 
# and stochastic gradient descent (SGD) for comparison
print('Using SGD: Stochastic Gradient Descent')
bsl_options1 = {'method': 'sgd',
               'learning_rate': 0.00005}

algo1 = BaselineOnly(bsl_options=bsl_options1) # Note algo1: SGD
print('==========================================')

print('Using ALS: Alternating Least Squares')
bsl_options = {'method':'als', 'n_epochs':5, 'regu_u': 12, 'reg_i':5}

algo = BaselineOnly(bsl_options=bsl_options) # Note algo: ALS

print('Cross validate SGD')
print('==========================================')
cross_validate(algo1, data, measures=['RMSE'], cv=5, verbose=False)

# Alternating Least Squares returns lower array of test RMSE
print('Cross validate ALS')
print('==========================================')
cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=False)

# Train, test, split to generate ratings predictions of each book
trainset, testset = train_test_split(data, test_size=0.20)
algo = BaselineOnly(bsl_options=bsl_options)
predictions = algo.fit(trainset).test(testset)
accuracy.rmse(predictions)

'''
* MAKING PREDICTIONS *
'''
print('Making predictions with book titles')

def get_Iu(uid):
    """ Return the number of items rated by given user
    Args: 
      uid: the id of the user
    Returns: 
      the number of items rated by the user
    """
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError: # user was not part of the trainset
        return 0
    
def get_Ui(iid):
    """ Return number of users that have rated given item
    Args:
      iid: the raw id of the item
    Returns:
      the number of users that have rated the item.
    """
    try: 
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:
        return 0

df_pred = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
df_pred['Iu']=df_pred['uid'].apply(get_Iu)
df_pred['Ui']=df_pred['iid'].apply(get_Ui)
df_pred['err']=abs(df_pred.est - df_pred.rui)

print('Shape of df_pred:', df_pred.shape)
print('==========================================')
print('')
print('Predictions Dataframe:')
print('==========================================')
print('')
df_pred.head()

# Top 20 Best Predictions
print('''
uid - user id
iid - isbn
rui - actual user rating
est - predicted ratings
Iu -  number of items rated by given user
Ui - number of users that have rated given item
''')
print('')
print('Here are the predicted top 20 books by ISBN numbers:')
best_predictions = df_pred.sort_values(by='err', ascending=True)[:20]
best_predictions.sort_values(by='Ui', ascending=False)

# Top 20 Worst Predictions
print('')
print('Here are the predicted top 20 books by ISBN numbers:')
print('''
uid - user id
iid - isbn
rui - actual user rating
est - predicted ratings
Iu -  number of items rated by given user
Ui - number of users that have rated given item
''')
worst_predictions = df_pred.sort_values(by='err', ascending=False).head(20)
worst_predictions
# worst_predictions.sort_values(by='Ui', ascending=False)

'''
* Top Recommendations for Users with Book Title *
'''

def get_top3_recommendations(predictions, topN = 3):
     '''
     Generations top three book recommendations by user-id.
     Parameters:
        - predictions made by algorithm
        - topN = 3 by default.
     Returns:
     '''
    top_recs = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_recs[uid].append((iid, est))
     
    for uid, user_ratings in top_recs.items():
        user_ratings.sort(key = lambda x: x[1], reverse = True)
        top_recs[uid] = user_ratings[:topN]
     
    return top_recs

def top3_book_titles(top_predictions):
    '''
    Generates top 3 book titles for each user based on the
    predictions generated by the best performing algorithm.
    Parameter: top_predictions
    Returns: Top 3 Book titles
    '''
    book_list = []
    for uid, user_ratings in top_predictions.items():
        book_list.append([uid, [iid for (iid, _) in user_ratings]])
    book_list_df=pd.DataFrame(book_list)
    book_list_df.rename(columns={0:'User-ID', 1:'ISBN'}, inplace=True)

    # First random five users
    first_five = book_list_df.head()
    first_five[['Book1', 'Book2','Book3']] = pd.DataFrame(first_five.ISBN.values.tolist(), index=first_five.index)
    first_five.drop('ISBN',1,inplace=True)
    books = ['Book1','Book2','Book3']
    for i in books:
        first_five=pd.merge(first_five, books_ratings_df[['ISBN', 'Book-Title']],
                        left_on=i, right_on='ISBN', how='left')
    rec_df = first_five.drop_duplicates()
    print('Top 3 Recommended Books for First 5 Random Users:')
    print('-------------------------------------------------')
    return rec_df[['User-ID', 'Book-Title_x', 'Book-Title_y', 'Book-Title']]

# Make recommendations
top_preds=get_top3_recommendations(predictions, topN=3)
top3_book_titles(top_preds)
