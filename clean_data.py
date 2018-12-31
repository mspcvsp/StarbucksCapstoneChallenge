""" Creates an analytic dataset from the Starbucks Capstone challenge 
data files:

* portfolio.json - contains offer ids and meta data (duration, type, etc.)

    * id (string) - offer id
    * offer_type (string) - string that describes the offer type
    * difficulty (int) - minimum a customer has to spend to complete an offer
    * reward (int) - reward given for completing an offer
    * duration (int) - offer duration [days]
    * channels - list of strings that describe how an offer is communicated
                 to a customer

* profile.json - demographic data for each customer
    * age (int) - age of the customer 
    * became_member_on (int) - date when customer created an app account
    * gender (str) - gender of the customer (note some entries contain 
                    'O' for other rather than M or F)
    * id (str) - customer id
    * income (float) - customer's income

* transcript.json - records for transactions, offers received, offers viewed, 
                    and offers completed

    * event (str) - record description (i.e. transaction, offer received,
                    offer viewed, etc.)
    * person (str) - customer id
    * time (int) - time in hours. The data begins at time t=0
    * value - (dict of strings) - either an offer id or transaction amount
              depending on the record

References:
----------
https://github.com/jtleek/datasharing - Defines what an analytic dataset is

https://stackoverflow.com/questions/41968732/
    set-order-of-columns-in-pandas-dataframe

https://stackoverflow.com/questions/40245507/
    python-pandas-selecting-rows-whose-column-value-is-null-none-nan

https://scikit-learn.org/stable/modules/generated/
    sklearn.preprocessing.MultiLabelBinarizer.html
    #sklearn.preprocessing.MultiLabelBinarizer

https://stackoverflow.com/questions/26454649/
    python-round-up-to-the-nearest-ten

https://stackoverflow.com/questions/44596077/
    datetime-strptime-in-python

https://stackoverflow.com/questions/4953272/
    how-to-match-exact-multiple-strings-in-python

https://stackoverflow.com/questions/402504/
    how-to-determine-a-python-variables-type

https://stackoverflow.com/questions/49728421/
    pandas-dataframe-settingwithcopywarning-a-value-is-trying-to-be-
    set-on-a-copy

https://stackoverflow.com/questions/43515877/
    should-binary-features-be-one-hot-encoded

https://www.shanelynn.ie/select-pandas-dataframe-rows-and-
    columns-using-iloc-loc-and-ix/

https://stackoverflow.com/questions/38987/how-to-merge-two-
    dictionaries-in-a-single-expression

https://stackoverflow.com/questions/3002085/python-to-print-out-
    status-bar-and-percentage

https://progressbar-2.readthedocs.io/en/latest/progressbar.bar.html

https://progressbar-2.readthedocs.io/en/latest/index.html#introduction
"""

from datetime import datetime
import numpy as np
import pandas as pd
import re
import os
import progressbar
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer


def update_column_name(dataframe,
                       old_column_name,
                       new_column_name):
    """ Updates a Pandas DataFrame column name

    INPUT:
        dataframe: Pandas DataFrame object

        old_column_name: String that stores the old column name

        new_column_name: String that stores the new column name

    OUTPUT:
        column_names: np.array that stores the updated Pandas DataFrame
                      column names"""
    column_names = dataframe.columns.values
    
    select_data = np.array([elem == old_column_name for elem in column_names])

    column_names[select_data] = new_column_name
        
    return column_names


def clean_portfolio(data_dir="./data"):
    """ 
    Transforms a DataFrame containing offer ids and meta data about 
    each offer (duration, type, etc.)

    INPUT:
        (Optional) data_dir: String that stores the full path to the
                             data directory

    OUTPUT:
        portfolio: DataFrame containing offer ids and meta data about 
                   each offer (duration, type, etc.)
    """
    portfolio = pd.read_json(os.path.join(data_dir, 'portfolio.json'),
                             orient='records',
                             lines=True)
    
    # Change the name of the 'id' column to 'offerid'
    columns = update_column_name(portfolio,
                                 'id',
                                 'offerid')
    
    # Change the name of the 'duration' column to 'durationdays'
    portfolio.columns = update_column_name(portfolio,
                                           'duration',
                                           'durationdays')

    # Remove underscores from column names
    portfolio.columns = [re.sub('_', '', elem) for elem in columns]

    # Initialize a list that stores the desired output DataFrame 
    # column ordering
    column_ordering = ['offerid',
                       'difficulty',
                       'durationdays',
                       'reward']

    # One hot encode the 'offertype' column
    offertype_df = pd.get_dummies(portfolio['offertype'])

    column_ordering.extend(offertype_df.columns.values)

    # One hot encode the 'channels' columns
    ml_binarizerobj = MultiLabelBinarizer()
    ml_binarizerobj.fit(portfolio['channels'])

    channels_df =\
        pd.DataFrame(ml_binarizerobj.transform(portfolio['channels']),
        columns=ml_binarizerobj.classes_)

    column_ordering.extend(channels_df.columns.values)

    # Replace the 'offertype' and 'channels' columns
    portfolio = pd.concat([portfolio, offertype_df, channels_df], axis=1)

    portfolio = portfolio.drop(columns=['offertype', 'channels'])

    # Return the "cleaned" portfolio data
    return portfolio[column_ordering]


def convert_to_datetime(elem):
    """Converts a string to a datetime object
    
    INPUT:
        elem: String that stores a date in the %Y%m%d format

    OUTPUT:
        datetimeobj: Datetime object"""
    return datetime.strptime(str(elem), '%Y%m%d')


def clean_profile(data_dir = "./data"):
    """ Transforms a DataFrame that contains demographic data for each 
    customer
    
    INPUT:
        (Optional) data_dir: String that stores the full path to the
                             data directory
    
    OUTPUT:
        profile: DataFrame that contains demographic data for each 
                 customer
    """
    profile = pd.read_json('data/profile.json',
                           orient='records',
                           lines=True)

    # Remove customers with N/A income data
    profile = profile[profile['income'].notnull()]

    # Remove customers with unspecified gender
    profile = profile[profile['gender'] != 'O']
    profile = profile.reset_index(drop=True)

    # Change the name of the 'id' column to 'customerid'
    profile.columns = update_column_name(profile,
                                         'id',
                                         'customerid')

    # Initialize a list that describes the desired DataFrame column
    # ordering
    column_ordering = ['customerid',
                       'gender',
                       'income']

    # Transform the 'became_member_on' column to a datetime object
    profile['became_member_on'] =\
        profile['became_member_on'].apply(convert_to_datetime)

    # One hot encode a customer's membership start year
    profile['membershipstartyear'] =\
        profile['became_member_on'].apply(lambda elem: elem.year)

    membershipstartyear_df = pd.get_dummies(profile['membershipstartyear'])
    column_ordering.extend(membershipstartyear_df.columns.values)

    # One hot encode a customer's age range
    min_age_limit = np.int(np.floor(np.min(profile['age'])/10)*10)
    max_age_limit = np.int(np.ceil(np.max(profile['age'])/10)*10)

    profile['agerange'] =\
        pd.cut(profile['age'],
               (range(min_age_limit,max_age_limit + 10, 10)),
               right=False)

    profile['agerange'] = profile['agerange'].astype('str')

    agerange_df = pd.get_dummies(profile['agerange'])
    column_ordering.extend(agerange_df.columns.values)

    # Transform a customer's gender from a character to a number
    binarizerobj = LabelBinarizer()
    profile['gender'] = binarizerobj.fit_transform(profile['gender'])

    gender_integer_map = {}
    for elem in binarizerobj.classes_:
        gender_integer_map[elem] = binarizerobj.transform([elem])[0,0]

    # Appened one hot encoded age range and membership start year variables
    profile = pd.concat([profile,
                         agerange_df,
                         membershipstartyear_df], axis=1)

    # Drop depcreated columns
    profile = profile.drop(columns=['age',
                                    'agerange',
                                    'became_member_on',
                                    'membershipstartyear'])

    # Return a DataFrame with "clean" customer profile data
    return profile[column_ordering], gender_integer_map

def clean_transcript(profile,
                     data_dir = './data'):
    """ Transforms a DataFrame that contains records for transactions, offers
    received, offers viewed, and offers completed

    INPUT:
        profile: DataFrame that contains demographic data for each 
                 customer

        (Optional) data_dir: String that stores the full path to the
                             data directory

    OUTPUT:
        offer_data: DataFrame that describes customer offer data

        transaction: DataFrame that describes customer transactions
    """
    transcript = pd.read_json(os.path.join(data_dir,
                                           'transcript.json'),
                              orient='records',
                              lines=True)

    # Change the name of the 'person' column to 'customerid'
    transcript.columns = update_column_name(transcript,
                                            'person',
                                            'customerid')

    # Remove customer id's that are not in the customer profile DataFrame
    select_data = transcript['customerid'].isin(profile['customerid'])
    transcript = transcript[select_data]

    percent_removed = 100 * (1 - select_data.sum() / select_data.shape[0])
    print("Percentage of transactions removed: %.2f %%" % percent_removed)

    # Convert from hours to days
    transcript['time'] /= 24.0
    
    # Change the name of the 'time' column to 'timedays'
    transcript.columns = update_column_name(transcript,
                                            'time',
                                            'timedays')

    # Select customer offers
    pattern_obj = re.compile('^offer (?:received|viewed|completed)')

    h_is_offer = lambda elem: pattern_obj.match(elem) != None

    is_offer = transcript['event'].apply(h_is_offer)

    offer_data = transcript[is_offer].copy()
    offer_data = offer_data.reset_index(drop=True)

    # Initialize a list that describes the desired output DataFrame
    # column ordering
    column_order = ['offerid', 'customerid', 'timedays']

    # Create an offerid column
    offer_data['offerid'] =\
        offer_data['value'].apply(lambda elem: list(elem.values())[0])

    # Transform a column that describes a customer offer event
    pattern_obj = re.compile('^offer ([a-z]+$)')

    h_transform = lambda elem: pattern_obj.match(elem).groups(1)[0]

    offer_data['event'] = offer_data['event'].apply(h_transform)

    # One hot encode customer offer events
    event_df = pd.get_dummies(offer_data['event'])
    column_order.extend(event_df.columns.values)

    # Create a DataFrame that describes customer offer events
    offer_data = pd.concat([offer_data, event_df], axis=1)
    offer_data.drop(columns=['event', 'value'])
    offer_data = offer_data[column_order]

    # Select customer transaction events
    transaction = transcript[is_offer == False]
    transaction = transaction.reset_index(drop=True)

    # Transform customer transaction event values
    transaction['amount'] =\
        transaction['value'].apply(lambda elem: list(elem.values())[0])

    # Create a DataFrame that describes customer transactions
    transaction = transaction.drop(columns=['event', 'value'])
    column_order = ['customerid', 'timedays', 'amount']
    transaction = transaction[column_order]

    return offer_data, transaction

def create_offeranalysis_dataset(profile,
                                 portfolio,
                                 offer_data,
                                 transaction):
    """ Creates an analytic dataset from the following Starbucks challenge 
    datasets:

    * portfolio.json - Contains offer ids and meta data (duration, type,
                       etc.)

    * profile.json - demographic data for each customer

    * transcript.json - records for transactions, offers received, offers
                        viewed, and offers completed
                        
    INPUT:
        profile: DataFrame that contains demographic data for each 
                 customer

        portfolio: Contains offer ids and meta data (duration, type, etc.)

        offer_data: DataFrame that describes customer offer data

        transaction: DataFrame that describes customer transactions

    OUTPUT:
        clean_data: DataFrame that characterizes the effectiveness of
                    customer offers"""
    clean_data = []
    customerid_list = offer_data['customerid'].unique()

    widgets=[' [',
             progressbar.Timer(), '] ',
             progressbar.Bar(),
             ' (',
             progressbar.ETA(),
             ') ']

    for idx in progressbar.progressbar(range(len(customerid_list)),
                                       widgets=widgets):

        clean_data.extend(create_combined_records(customerid_list[idx],
                                                  portfolio,
                                                  profile,
                                                  offer_data,
                                                  transaction))

    clean_data = pd.DataFrame(clean_data)

    # Initialize a list that describes the desired output DataFrame
    # column ordering
    column_ordering = ['time', 'offerid', 'customerid', 'totalamount',
                       'offersuccessful', 'difficulty', 'durationdays',
                       'reward', 'bogo', 'discount', 'informational',
                       'email', 'mobile', 'social', 'web', 'gender',
                       'income', 2013, 2014, 2015, 2016, 2017, 2018,
                       '[10, 20)', '[20, 30)', '[30, 40)', '[40, 50)',
                       '[50, 60)', '[60, 70)', '[70, 80)', '[80, 90)',
                       '[90, 100)', '[100, 110)']

    clean_data = clean_data[column_ordering]

    clean_data = clean_data.sort_values('time')
    return clean_data.reset_index(drop=True)

def create_combined_records(customer_id,
                            portfolio,
                            profile,
                            offer_data,
                            transaction):
    """ 
    Creates a list of dictionaries that describes the effectiveness of
    offers to a specific customer

    INPUT:
        customer_id: String that refers to a specific customer

        profile: DataFrame that contains demographic data for each 
                 customer
                 
        portfolio: DataFrame containing offer ids and meta data about 
                   each offer (duration, type, etc.)

        offer_data: DataFrame that describes customer offer data

        transaction: DataFrame that describes customer transactions
    
    OUTPUT:
        rows: List of dictionaries that describes the effectiveness of
              offers to a specific customer
    """
    # Select a customer's profile
    cur_customer = profile[profile['customerid'] == customer_id]

    # Select offer data for a specific customer
    select_offer_data = offer_data['customerid'] == customer_id
    customer_offer_data = offer_data[select_offer_data]
    customer_offer_data = customer_offer_data.drop(columns='customerid')
    customer_offer_data = customer_offer_data.reset_index(drop=True)

    # Select transactions for a specific customer
    select_transaction = transaction['customerid'] == customer_id
    customer_transaction_data = transaction[select_transaction]

    customer_transaction_data =\
        customer_transaction_data.drop(columns='customerid')

    customer_transaction_data =\
        customer_transaction_data.reset_index(drop=True)

    # Initialize DataFrames that describe when a customer receives,
    # views, and completes an offer
    event_type = ['completed',
                  'received',
                  'viewed']

    offer_received =\
        customer_offer_data[customer_offer_data['received'] == 1]

    offer_received = offer_received.drop(columns=event_type)
    offer_received = offer_received.reset_index(drop=True)

    offer_viewed =\
        customer_offer_data[customer_offer_data['viewed'] == 1]

    offer_viewed = offer_viewed.drop(columns=event_type)
    offer_viewed = offer_viewed.reset_index(drop=True)

    offer_completed =\
        customer_offer_data[customer_offer_data['completed'] == 1]

    offer_completed = offer_completed.drop(columns=event_type)
    offer_completed = offer_completed.reset_index(drop=True)

    # Iterate over each offer a customer receives
    rows = []
    for idx in range(offer_received.shape[0]):

        # Initialize the current offer id
        cur_offer_id = offer_received.iloc[idx]['offerid']

        # Look-up a description of the current offer
        cur_offer = portfolio.loc[portfolio['offerid'] == cur_offer_id]
        durationdays = cur_offer['durationdays'].values[0]

        # Initialize the time period when an offer is valid
        cur_offer_startime = offer_received.iloc[idx]['timedays']

        cur_offer_endtime =\
            offer_received.iloc[idx]['timedays'] + durationdays

        # Initialize a boolean array that select customer transcations that
        # fall within the valid offer time window
        select_transaction =\
            np.logical_and(customer_transaction_data['timedays'] >=
                           cur_offer_startime,
                           customer_transaction_data['timedays'] <=
                           cur_offer_endtime)

        # Initialize a boolean array that selects a description of when a
        # customer completes an offer (this array may not contain any True
        # values)
        select_offer_completed =\
            np.logical_and(offer_completed['timedays'] >= cur_offer_startime,
                           offer_completed['timedays'] <= cur_offer_endtime)

        # Initialize a boolean array that selects a description of when a
        # customer views an offer (this array may not contain any True
        # values)
        select_offer_viewed =\
            np.logical_and(offer_viewed['timedays'] >= cur_offer_startime,
                           offer_viewed['timedays'] <= cur_offer_endtime)

        # Determine whether the current offer was successful
        cur_offer_successful =\
            select_offer_completed.sum() > 0 and select_offer_viewed.sum() > 0

        # Select customer transcations that occurred within the current offer
        # valid time window
        cur_offer_transactions = customer_transaction_data[select_transaction]

        # Initialize a dictionary that describes the current customer offer
        cur_row = {'offerid': cur_offer_id,
                   'customerid': customer_id,
                   'time': cur_offer_startime,
                   'offersuccessful': int(cur_offer_successful),
                   'totalamount': cur_offer_transactions['amount'].sum()}

        cur_row.update(cur_offer.iloc[0,1:].to_dict())

        cur_row.update(cur_customer.iloc[0,1:].to_dict())

        # Update a list of dictionaries that describes the effectiveness of 
        # offers to a specific customer
        rows.append(cur_row)

    return rows
