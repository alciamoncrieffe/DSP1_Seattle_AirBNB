import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def date_to_months(date, comparitive_date):
    '''
    
    
    INPUT
        date - date
        comparitive_date - recent date to be compared to the old one 
    
    OUTPUT
        int - number of months between input date and comparitive_date
    
    '''
    return (comparitive_date.year - date.year) * 12 + comparitive_date.month - date.month


def multihost(listing_count):
    '''
    Input : listing_count - int (count of properties owned by listing host)
    Output : t/f - string (true or false whether the host owns more than 1 property)
    '''
    if listing_count > 1:
        return 't'
    else:
        return 'f'


def calendar_updated_date(time_string, comparitive_date, host_since):
    '''
    INPUT
        time_string - list of tokenized calendar_updated string
        comparitive_date - recent date to be compared to the old one 
        host_since - date of start of hosting
    
    OUTPUT
        date - calendar_updated string converted to a date
    
    '''
    unit_value = 0
    
    if str(time_string[0]) == 'a':
        unit_value = 1
        
    if str(time_string[0]).isnumeric():
        unit_value = int(time_string[0])
    
    if unit_value != 0:
        if str(time_string[1]) in ['days','day']:
            return comparitive_date - relativedelta(days=+unit_value)
        elif str(time_string[1]) in ['weeks','week']:
            return comparitive_date - relativedelta(weeks=+unit_value)
        elif str(time_string[1]) in ['months','month']:
            return comparitive_date - relativedelta(months=+unit_value)
        else:
            return time_string
    elif str(time_string[0]) == 'today':
        return comparitive_date
    elif str(time_string[0]) == 'yesterday':
        return comparitive_date - relativedelta(days=1)
    else:
        return host_since

    
def has_value(null):
    '''
    INPUT
        pic - string of picture url
    
    OUTPUT
        t or f - string corresponding to whether or not pic is null
    '''
    if null:
        return 'f'
    else:
        return 't'

    
def summerize_columns(df):
    '''
    Summarises the listings_df columns into a new dataframe. 
    Adding the count of non-null values, column population percentage, number of unique values per column and data type.
    
    INPUT
        df - dataframe
    
    OUTPUT
        summary_info - a dataframe containing the column names of the input dataframe including a count of values per column, 
            percentage filled, distinct value count and column type
    '''
    summary_info = []

    for col in df.columns:

        nonNull  = len(df) - np.sum(pd.isna(df[col]))
        pop_per = nonNull / df.shape[0]
        unique = df[col].nunique()
        colType = str(df[col].dtype)

        summary_info.append([col, nonNull, pop_per, unique, colType])

    summary_info = pd.DataFrame(summary_info)   
    summary_info.columns = ['colName','non-null values', 'pop percentage', 'unique', 'dtype']
    return summary_info


def linear_model(df, response_col, explanatory_cols, test_size, random_state):
    '''
    INPUT
        df - a dataframe holding all the variables of interest
        response_col - a string holding the name of the column 
        explanatory_cols - list of strings that are associated with names of the chosen explanatory. 
                Can be null. Then the default will be to use all columns for X
        test_size - a float between [0,1] about what proportion of data should be in the test dataset
        rand_state - an int that is provided as the random state for splitting the data into training and test
    
    OUTPUT
        test_score - float - r2 score on the test data
        train_score - float - r2 score on the test data
        lm_model - model object from sklearn
        X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    #Split into explanatory and response variables
    if not explanatory_cols:
        X = df.drop(response_col, axis=1)
    else:
        X = df[explanatory_cols]
    y = df[response_col]

    #Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    lm_model = LinearRegression(normalize=True) # Instantiate
    lm_model.fit(X_train, y_train) #Fit

    #Predict using your model
    y_test_preds = lm_model.predict(X_test)
    y_train_preds = lm_model.predict(X_train)

    #Score using your model
    test_score = r2_score(y_test, y_test_preds)
    train_score = r2_score(y_train, y_train_preds)

    return test_score, train_score, lm_model, X_train, X_test, y_train, y_test 

def word_count(df, text_col, limit_no):
    '''
    INPUT
        df - dataframe
        text_col - string. Column name of a free text string column
        limit_no - int. Number theshold to limit the number of rare words added to the word_dict 
    
    OUTPUT
        word_dict - dataframe made up of a dictionary of words and a count of the number of times they appeared in the df column
    '''
    #nltk_words are a list of common words that will be ignored when examining the text and not added to the final df
    nltk_words = ['','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    
    df[text_col] = df[text_col].str.lower() 
    df[text_col] = df[text_col].str.replace(',','',regex=True)
    df[text_col] = df[text_col].str.replace('.','',regex=True)
    df[text_col] = df[text_col].str.replace('!','',regex=True)
    df[text_col] = df[text_col].str.replace('"','',regex=True)
    df[text_col] = df[text_col].str.replace('/','',regex=True)
    df[text_col] = df[text_col].str.replace('?','',regex=True)
    df[text_col] = df[text_col].str.replace('(','',regex=True)
    df[text_col] = df[text_col].str.replace(')','',regex=True)
    df[text_col] = df[text_col].str.replace(':','',regex=True)
    df[text_col] = df[text_col].str.replace(';','',regex=True)
    df[text_col] = df[text_col].str.split(' ')

    word_set = dict()
    for record in df[text_col]:
        try:
            for word in record: 
                if (word not in nltk_words): #If not in the list of common words
                    if word not in word_set :#and not in the new word dictionary 
                        word_set[word] = 1   #then add the word to the dictionary
                    else:
                        word_set[word] += 1
        except:
            0    
    #new dictionary created based on words with a count >= input number limit
    new_dict = {word : word_set[word] for word in word_set.keys() if word_set[word] >= limit_no} 
    return pd.DataFrame.from_dict(new_dict, orient='index', columns=[ 'count'])