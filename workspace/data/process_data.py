# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3



def load_data(messages_filepath, categories_filepath):
    
    """
       Function:
           Loads the messages and categories datasets and then merges them into a single dataframe
       Args:
         messages_filepath: Filepath to the messages dataset
         categories_filepath: Filepath to the categories dataset
       Returns:
         Merged Pandas dataframe
    """

    messages = pd.read_csv(messages_filepath)  # load messages dataset
    categories = pd.read_csv(categories_filepath)  # load categories dataset
    df = messages.merge(categories, on='id')  # merge datasets
    
    return df


def clean_data(df):
    
    """
       Function:
           Cleans the merged dataset
       Args:
           Merged pandas dataframe
       Returns:
           Cleaned dataframe
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.head(1)  # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = categories.applymap(lambda x: x[:-2]).iloc[0, :]
    category_colnames = category_colnames.tolist()
    categories.columns = category_colnames  # rename the columns of `categories`

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df.drop('categories', axis=1)  # drop the original categories column from `df`
    # concatenate the original dataframe with the new `categories` dataframe
    df = df = pd.concat([df, categories], axis=1, join='inner')
    df.drop_duplicates(inplace=True)  # drop duplicates
    df = df[df['related'] != 2]

    return df


def save_data(df, database_filename):
    
    """
       Function:
           Save the Dataframe in a database
       Args:
           A dataframe of both messages and categories files
           database_filename for the databas
       Returns:
          None
     """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    # Save the clean dataset into an sqlite database
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()