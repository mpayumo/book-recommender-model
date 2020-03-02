'''
This model recommends the top rated book titles
and runs a pearson correlation book recommendation 
system as a starter model.
'''

# Import packages and modules
from src.utilities import *
import pandas as pd
import numpy as np

# Read CSVs
ratings_df = csv('https://markg110.s3-us-west-1.amazonaws.com/data/BX-Book-Ratings.csv')
books_df = csv('https://markg110.s3-us-west-1.amazonaws.com/data/BX-Books.csv')
users_df = csv('https://markg110.s3-us-west-1.amazonaws.com/data/BX-Users.csv')

# Explore shapes of dataframes
print('ratings_df: ', ratings_df.shape)
print('users_df: ', users_df.shape)
print('books_df: ', books_df.shape)

# Exploring null values
print(ratings_df.isna().sum())
print('')
print(users_df.isna().sum())
print('')
print(books_df.isna().sum())

# Dropping unnecessary columns
columns = ['Publisher', 'Book-Author', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
books_df.drop(columns=columns, inplace=True)

# Merging ratings and books dataframes in order to reconcile ratings
books_ratings_df = pd.merge(ratings_df, books_df, on='ISBN')

# Minor data cleaning
books_ratings_df = books_ratings_df[books_ratings_df['Publication-Year'] != 'Gallimard']
books_ratings_df = books_ratings_df[books_ratings_df['Publication-Year'] != 'DK Publishing Inc']
books_ratings_df['Publication-Year'] = books_ratings_df['Publication-Year'].apply(lambda x: int(x))

'''
1>> Recommendations Based on Most Rated Books
Total and average number of ratings
'''
average_count_rating = pd.DataFrame(books_ratings_df.groupby('ISBN')['Book-Rating'].mean())
average_count_rating['Rating-Count'] = pd.DataFrame(books_ratings_df.groupby('ISBN')['Book-Rating'].count())
average_count_rating.rename(columns={'Book-Rating':'Average-Rating'},inplace=True)

# Top 10 Most Rated Movies
test10=average_count_rating.sort_values(by='Rating-Count', ascending=False).head(10)
top_isbn=test10.index.to_list()
most_rated = pd.DataFrame(top_isbn, index=np.arange(len(top_isbn)), columns=['ISBN'])
most_rated_books = pd.merge(most_rated, books_ratings_df, on='ISBN')
top_book_titles = most_rated_books['Book-Title'].value_counts().to_frame()
top_book_titles.rename(columns={'Book-Title':'Count'},inplace=True)

# Top 10 Most Rated Movie Titles
# pd.Series(top_book_titles.index)

# 2>> Pearson Correlation Recommender
'''
In order to maintain statistical significance in our results, 
we will limit the dataset to at least 200 ratings given by users
and at least 100 ratings received by books.
'''

# Trimming down dataset
counts1 = ratings_df['User-ID'].value_counts()
counts2 = ratings_df['Book-Rating'].value_counts()
ratings = ratings_df[ratings_df['User-ID'].isin(counts1[counts1 >= 200].index)] 
ratings = ratings[ratings['Book-Rating'].isin(counts2[counts2 >= 100].index)]

# User-item matrix to calculate correlation
book_matrix = pd.pivot_table(ratings, index='User-ID', columns='ISBN', values='Book-Rating')

def recommend_books(ratings_matrix, average_ratings_rating_count, book_matrix_isbn_column):
    '''
    ===============================================================================
    Recommends book titles with highest positive correlation to ISBN input number.
    -------------------------------------------------------------------------------
    Parameters :

    ratings_matrix : User-item ratings matrix from following execution of pd.pivot_table function
    average_ratings_rating_count : In format, average_ratings['Rating-Count']
    book_matrix_isbn_column : In format, book_matrix['12345678']. NOTE: ISBN is type string.

    ===============================================================================
    '''
    user_rating = book_matrix_isbn_column
    similar_to_user_rating = ratings_matrix.corrwith(user_rating)
    correlated_books = pd.DataFrame(similar_to_user_rating, columns=['pearsonR'])
    correlated_books.dropna(inplace=True)
    correlation_summary = correlated_books.join(average_ratings_rating_count)
    isbn_only = correlation_summary[correlated_summary['Rating-Count'] >= 300].sort_values('pearsonR', ascending=False).head(11)
    isbn_list=isbn_only.index.tolist()
    books_isbn = pd.DataFrame(isbn_list, index=np.arange(len(isbn_list)), columns=['ISBN'])
    correlated_book_titles=pd.merge(books_isbn, books_df, on='ISBN')
    return correlated_book_titles

print('===============================================================================')
print('')
print('The Top 10 Most Rated Movie Titles:')
pd.Series(top_book_titles.index)
print('===============================================================================')
print('')
print('The Top Recommended Movies Based on Pearson R:')
print('')
recommend_books(book_matrix, average_count_rating['Rating-Count'], book_matrix['0316666343'])





