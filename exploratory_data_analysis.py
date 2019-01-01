"""Exploratory data analysis functions

References:
----------
https://stackoverflow.com/questions/38334296/
    reversing-one-hot-encoding-in-pandas

https://stackoverflow.com/questions/4406389/if-else-in-a-list-comprehension

https://stackoverflow.com/questions/17679089/
    pandas-dataframe-groupby-two-columns-and-get-counts

https://stackoverflow.com/questions/10373660/
    converting-a-pandas-groupby-object-to-dataframe
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns


def reverse_ohe(dataframe_row,
                is_ohe_column):
    """ Reverses categorical one hot encoding

    INPUT:
        dataframe_row: Pandas Series that stores a DataFrame row

        is_ohe_column: List that describes which DataFrame columns
                       correspond to the one hot encoding of a 
                       categorical variable

    OUTPUT:
        categorical_variable: String that stores the value of a
                              categorical variable"""
    column_idx = np.argwhere(dataframe_row[is_ohe_column])

    return str(dataframe_row.index.values[is_ohe_column][column_idx][0,0])

def genderint_to_string(dataframe,
                        gender_integer_map):
    """Transforms gender encoded as an integer to a string

    INPUT:
        dataframe: DataFrame that includes gender encoded as an integer

        gender_integer_map: Dictionary that describes the mapping of a
                            gender string to an integer

    OUTPUT:
        gender: Series that encodes gender as a string"""
    gender = ['Male' if elem == 'M' else 'Female'
              for elem in gender_integer_map.keys()]

    integer_gender_map = dict(zip(gender_integer_map.values(), gender))

    h_transfrom = lambda elem: integer_gender_map[elem]

    return dataframe['gender'].apply(h_transfrom)

def initialize_membership_date(profile,
                               gender_integer_map):
    """ Initializes a DataFrame that describes the membership start year
        statistics as a function of customer gender

        INPUT:
            profile: DataFrame that stores customer profiles

            gender_integer_map: Dictionary that describes the mapping of a
                                gender string to an integer

        OUTPUT:
            membership_date: DataFrame that describes the membership start
                             year statistics as a function of customer 
                             gender"""
    is_ohe_column = [type(elem) == int for elem in profile.columns]

    membership_date = profile[['gender']].copy()

    membership_date['gender'] = genderint_to_string(membership_date,
                                                    gender_integer_map)

    membership_date['startyear'] =\
        profile.apply(lambda elem: reverse_ohe(elem, is_ohe_column), axis=1)

    membership_date = membership_date.groupby(['startyear', 'gender']).size()

    membership_date = membership_date.reset_index()

    membership_date.columns = ['startyear', 'gender', 'count']
    
    return membership_date

def init_agerange(profile,
                  gender_integer_map):
    """ Initializes a DataFrame that describes a customer's age range as 
    a function of gender
    
    INPUT:
        profile: DataFrame that stores customer profiles

        gender_integer_map: Dictionary that describes the mapping of a
                            gender string to an integer

    OUTPUT:
        age_range: DataFrame that describes a customer's age range as a
                   function of gender
    """
    pattern_obj = re.compile('^\[[0-9]+, [0-9]+\)')

    is_ohe_column =\
        [pattern_obj.match(str(elem)) != None for elem in profile.columns]

    age_range = profile[['gender']].copy()
    age_range['gender'] = genderint_to_string(age_range,
                                            gender_integer_map)

    age_range['agerange'] =\
        profile.apply(lambda elem: reverse_ohe(elem, is_ohe_column), axis=1)

    age_range = age_range.groupby(['agerange', 'gender']).size()

    age_range = age_range.reset_index()

    age_range.columns = ['agerange', 'gender', 'count']

    is_above_100 = re.compile('\[10[0-9]+.*')
    h_transform = lambda elem: is_above_100.match(elem) != None
    above_100 = age_range['agerange'].apply(h_transform)

    return pd.concat([age_range[above_100 == False], age_range[above_100]])


def initialize_percent_success(portfolio,
                               clean_data):
    """ Initializes a DataFrame that describes the success percentage for
    each offer
    
    INPUT:
        portfolio: DataFrame containing offer ids and meta data about 
                   each offer (duration, type, etc.)

        clean_data: DataFrame that characterizes the effectiveness of
                    customer offers
    
    OUTPUT:
        percent_success: DataFrame that describes the success percentage for
                         each offer"""
    successful_count =\
        clean_data[['offerid',
                    'offersuccessful']].groupby('offerid').sum().reset_index()

    offer_count = clean_data['offerid'].value_counts()

    offer_count = pd.DataFrame(list(zip(offer_count.index.values,
                                        offer_count.values)),
                               columns=['offerid', 'count'])

    successful_count = successful_count.sort_values('offerid')

    offer_count = offer_count.sort_values('offerid')

    percent_success = pd.merge(offer_count, successful_count, on="offerid")

    percent_success['percentsuccess'] =\
        100 * percent_success['offersuccessful'] / percent_success['count']

    percent_success = pd.merge(percent_success,
                               portfolio,
                               on="offerid")

    percent_success = percent_success.drop(columns=['offersuccessful'])

    percent_success = percent_success.sort_values('percentsuccess',
                                                  ascending=False)

    return percent_success.reset_index(drop=True)

def select_customer_data(offerid,
                         training_data,
                         gender_integer_map):
    """ Returns a DataFrame that stores combined transaction, demographic and
    offer data for a specific offer
    
    INPUT:
        offerid: String that refers to a specific offer
        
        training_data: DataFrame that stores combined transaction, 
                       demographic and offer data

        gender_integer_map: Dictionary that describes the mapping of a
                            gender string to an integer

    OUTPUT:
        customer_income: DataFrame that describes customer income statistics
                         associated with a specific offer

        customer_gender: DataFrame that describes customer gender statistics
                         associated with a particular offer

        customer_agerange: DataFrame that describes customer agerange
                           statistics associated with a particular offer

        membershipstartyear: DataFrame that describes customer membership
                             start year statistics associated with a
                             particular offer
    """

    # Segment offer data
    select_data = training_data['offerid'] == offerid
    cur_offer_data = training_data[select_data].copy()

    # Segment successful and unsuccessful offer data
    offer_successful = cur_offer_data['offersuccessful'].astype('bool')
    columns_to_drop = ['offerid', 'offersuccessful']
    cur_offer_data = cur_offer_data.drop(columns=columns_to_drop)

    successful_offer = cur_offer_data[offer_successful].copy()
    unsuccessful_offer = cur_offer_data[offer_successful == False].copy()

    # Reverse membership start year one hot encoding
    pattern_obj = re.compile('^20[0-9]{2}$')

    is_start_year = [pattern_obj.match(str(elem)) != None
                     for elem in successful_offer.columns.values]

    columns_to_drop = successful_offer.columns.values[is_start_year]

    h_transform = lambda elem: reverse_ohe(elem, is_start_year)

    successful_offer['membershipstartyear'] =\
        successful_offer.apply(h_transform, axis=1)

    unsuccessful_offer['membershipstartyear'] =\
        unsuccessful_offer.apply(h_transform, axis=1)

    successful_offer = successful_offer.drop(columns=columns_to_drop)
    unsuccessful_offer = unsuccessful_offer.drop(columns=columns_to_drop)

    # Reverse customer age range one hot encoding
    pattern_obj = re.compile('^\[[0-9]+, [0-9]+\)$')

    is_age_range = [pattern_obj.match(elem) !=
                    None for elem in successful_offer.columns]

    columns_to_drop = successful_offer.columns.values[is_age_range]

    h_transform = lambda elem: reverse_ohe(elem, is_age_range)

    successful_offer['agerange'] =\
        successful_offer.apply(h_transform, axis=1)

    unsuccessful_offer['agerange'] =\
        unsuccessful_offer.apply(h_transform, axis=1)

    successful_offer = successful_offer.drop(columns=columns_to_drop)
    unsuccessful_offer = unsuccessful_offer.drop(columns=columns_to_drop)

    # Reverse customer gender one hot encoding
    successful_offer['gender'] = genderint_to_string(successful_offer,
                                                     gender_integer_map)

    unsuccessful_offer['gender'] = genderint_to_string(unsuccessful_offer,
                                                       gender_integer_map)

    successful_offer = successful_offer.reset_index(drop=True)
    unsuccessful_offer = unsuccessful_offer.reset_index(drop=True)
    
    customer_income = initialize_customer_income(successful_offer,
                                                 unsuccessful_offer)

    customer_gender = initialize_customer_gender(successful_offer,
                                                 unsuccessful_offer)

    customer_agerange = initialize_customer_agerange(successful_offer,
                                                     unsuccessful_offer)

    membershipstartyear =\
        init_customer_membershipstartyear(successful_offer,
                                          unsuccessful_offer)

    return (customer_income,
            customer_gender,
            customer_agerange,
            membershipstartyear)

def initialize_customer_income(successful_offer,
                               unsuccessful_offer,
                               number_bins=30):
    """ Initializes a DataFrame that describes customer income statistics
    associated with a specific offer

    INPUT:
        successful_offer: DataFrame that describes customers that an offer
                          was successful for

        unsuccessful_offer: DataFrame that describes customers that an offer
                            was not successful for

        number_bins: (Optional) Number of histogram bins
    
    OUTPUT:
        customer_income: DataFrame that describes customer income statistics
                         associated with a specific offer
    """
    min_income = np.min([successful_offer['income'].min(),
                         unsuccessful_offer['income'].min()])

    max_income = np.max([successful_offer['income'].max(),
                         unsuccessful_offer['income'].max()])

    income_bins = np.linspace(min_income, max_income, number_bins)

    income_histogram = np.histogram(successful_offer['income'],
                                    income_bins,
                                    density=True)[0]

    column_names = ['income', 'probability', 'successful']

    successful_offer_income =\
        pd.DataFrame(list(zip(income_bins,
                              income_histogram,
                              np.repeat('Yes', number_bins))),
                              columns=column_names)

    income_histogram = np.histogram(unsuccessful_offer['income'],
                                    income_bins,
                                    density=True)[0]

    unsuccessful_offer_income =\
        pd.DataFrame(list(zip(income_bins,
                              income_histogram,
                              np.repeat('No', number_bins))),
                    columns=column_names)

    customer_income = pd.concat([successful_offer_income,
                                 unsuccessful_offer_income],
                                axis=0)

    return customer_income.reset_index(drop=True)

def initialize_customer_gender(successful_offer,
                               unsuccessful_offer):
    """ Initializes a DataFrame that describes customer gender statistics
    associated with a particular offer

    INPUT:
        successful_offer: DataFrame that describes customers that an offer
                          was successful for

        unsuccessful_offer: DataFrame that describes customers that an offer
                            was not successful for

    OUTPUT:
        customer_gender: DataFrame that describes customer gender statistics
                         associated with a particular offer
    """
    column_names = ['gender', 'percentage', 'successful']

    gender_histogram = successful_offer['gender'].value_counts()
    gender_histogram *= 100 / gender_histogram.sum()

    successful_offer_gender =\
        pd.DataFrame(list(zip(gender_histogram.index.values,
                              gender_histogram,
                              np.repeat('Yes', 2))),
                     columns=column_names)

    gender_histogram = unsuccessful_offer['gender'].value_counts()
    gender_histogram *= 100 / gender_histogram.sum()

    unsuccessful_offer_gender =\
        pd.DataFrame(list(zip(gender_histogram.index.values,
                              gender_histogram,
                              np.repeat('No', 2))),
                              columns=column_names)

    customer_gender = pd.concat([successful_offer_gender,
                                 unsuccessful_offer_gender],
                                axis=0)

    return customer_gender.reset_index(drop=True)

def initialize_agerange_histogram(dataframe):
    """ Initializes a DataFrame that describes customer age range statistics
    associated with a particular offer

    INPUT:
        dataframe: DataFrame that describes customer statistics associated
                   with a particular offer
    
    OUTPUT:
        agerange_histogram: DataFrame that describes customer age range
                            statistics associated with a particular offer

    References:
    ---------
    https://stackoverflow.com/questions/32015669/
        change-order-of-columns-in-stacked-bar-plot
    """
    agerange_histogram = dataframe['agerange'].value_counts()
    agerange_histogram *= 100 / agerange_histogram.sum()

    pattern_obj = re.compile('^\[([0-9]+), ([0-9]+)\)$')

    bin_start = [float(pattern_obj.match(elem).group(1))
                 for elem in agerange_histogram.index.values]

    bin_end = [float(pattern_obj.match(elem).group(2))
               for elem in agerange_histogram.index.values]

    bin_center = [np.mean(elem) for elem in zip(bin_start, bin_end)]

    sorted_agerange_idx = np.argsort(bin_center)

    return agerange_histogram.iloc[sorted_agerange_idx]

def initialize_customer_agerange(successful_offer,
                                 unsuccessful_offer):
    """ Initializes a DataFrame that describes customer agerange statistics
    associated with a particular offer
    
    INPUT:
        successful_offer: DataFrame that describes customers that an offer
                          was successful for

        unsuccessful_offer: DataFrame that describes customers that an offer
                            was not successful for

    OUTPUT:
        customer_agerange: DataFrame that describes customer agerange
                           statistics associated with a particular offer
    """
    column_names = ['agerange', 'percentage', 'successful']

    agerange_histogram = initialize_agerange_histogram(successful_offer)

    successful_agerange =\
        pd.DataFrame(list(zip(agerange_histogram.index.values,
                              agerange_histogram,
                              np.repeat('Yes', agerange_histogram.shape[0]))),
                     columns=column_names)

    agerange_histogram = initialize_agerange_histogram(unsuccessful_offer)

    unsuccessful_agerange =\
        pd.DataFrame(list(zip(agerange_histogram.index.values,
                              agerange_histogram,
                              np.repeat('No', agerange_histogram.shape[0]))),
                     columns=column_names)

    customer_agerange = pd.concat([successful_agerange,
                                   unsuccessful_agerange],
                                  axis=0)

    return customer_agerange.reset_index(drop=True)

def init_customer_membershipstartyear(successful_offer,
                                      unsuccessful_offer):
    """
    Initializes a DataFrame that describes customer membership start year
    statistics associated with a particular offer

    INPUT:
        successful_offer: DataFrame that describes customers that an offer
                          was successful for

        unsuccessful_offer: DataFrame that describes customers that an offer
                            was not successful for

    OUTPUT:
        membershipstartyear: DataFrame that describes customer membership
                             start year statistics associated with a
                             particular offer
    """
    successful_membershipstartyear =\
        init_membershipstartyear_histogram(successful_offer)
    
    unsuccessful_membershipstartyear =\
        init_membershipstartyear_histogram(unsuccessful_offer,
                                           'No')

    customer_membershipstartyear =\
        pd.concat([successful_membershipstartyear,
                   unsuccessful_membershipstartyear],
                  axis=0)

    return customer_membershipstartyear.reset_index(drop=True)
    
def init_membershipstartyear_histogram(dataframe,
                                       successful='Yes'):
    """ Initializes a DataFrame that describes customer membership start
    year statistics

    INPUT:
        dataframe: DataFrame that describes customer statistics

        successful: (Optional) String that describes whether an offer was
                    successful

    OUTPUT:
        histogram: DataFrame that describes customer membership start
                   year statistics"""
    histogram = dataframe['membershipstartyear'].value_counts()

    histogram *= 100 / histogram.sum()

    number_years = histogram.shape[0]

    histogram = pd.DataFrame(list(zip(histogram.index.values,
                                      histogram,
                                      np.repeat(successful, number_years))),
                             columns=['membershipstartyear',
                                      'percentage',
                                      'successful'])

    histogram['membershipstartyear'] =\
        histogram['membershipstartyear'].astype('int')

    return histogram.sort_values('membershipstartyear')

def print_customer_statistics(cur_offer_id,
                              customer_income,
                              customer_agerange):
    """Prints average customer income and age
    
    INPUT:
        cur_offer_id: String that refers to a specific offer

        customer_income: DataFrame that describes customer income statistics
                         associated with a specific offer

        customer_agerange: DataFrame that describes customer agerange
                           statistics associated with a particular offer

    OUTPUT:
        None
    """
    print("offerid: %s" % (cur_offer_id))
    
    print("Offer successful")

    print("\tAverage customer income: $%.1f" %\
          compute_average_income(customer_income))

    print("\tAverage customer age: %.1f [years]" %\
          compute_average_age(customer_agerange))

    print("Offer unsuccessful")

    print("\tAverage customer income: $%.1f" %\
          compute_average_income(customer_income, 'No'))
    
    print("\tAverage customer age: %.1f [years]" %\
          compute_average_age(customer_agerange, 'No'))
    
def compute_average_income(customer_income,
                           successful='Yes'):
    """ Computes the average customer income

    INPUT:
        customer_income: DataFrame that describes customer income statistics
                         associated with a specific offer

        successful: (Optional) String that describes whether an offer was
                    successful
    
    OUTPUT:
        average_income: Average customer income
    """
    select_data = customer_income['successful'] == successful

    cur_income = customer_income[select_data]['income']
    cur_probability = customer_income[select_data]['probability']
    cur_income_weighting = cur_probability / np.sum(cur_probability)

    return np.sum(cur_income * cur_income_weighting)

def compute_average_age(customer_agerange,
                        successful='Yes'):
    """ Computes the average customer age

    INPUT:
        customer_agerange: DataFrame that describes customer agerange
                           statistics associated with a particular offer

        successful: (Optional) String that describes whether an offer was
                    successful

    OUTPUT:
        average_age: Average customer age
    """
    select_data = customer_agerange['successful'] == successful
    
    cur_agerange = customer_agerange[select_data]['agerange']
    cur_percentage = customer_agerange[select_data]['percentage']

    pattern_obj = re.compile('^\[([0-9]+), ([0-9]+)\)$')

    min_age = [int(pattern_obj.match(elem).group(1)) for elem in cur_agerange]
    max_age = [int(pattern_obj.match(elem).group(2)) for elem in cur_agerange]
    avg_age = [np.mean(elem) for elem in zip(min_age, max_age)]

    return np.sum(avg_age * cur_percentage / 100)

def visualize_customer_statistics(cur_offer_id,
                                  customer_income,
                                  customer_gender,
                                  customer_agerange,
                                  membershipstartyear):
    """
    Visualizes customer statistics for a specific offer

    INPUT:
        customer_income: DataFrame that describes customer income statistics
                         associated with a specific offer

        customer_gender: DataFrame that describes customer gender statistics
                         associated with a particular offer

        customer_agerange: DataFrame that describes customer agerange
                           statistics associated with a particular offer

        membershipstartyear: DataFrame that describes customer membership
                             start year statistics associated with a
                             particular offer

    OUTPUT:
        None

    Reference:
    ---------
    https://stackoverflow.com/questions/31859285/
        rotate-tick-labels-for-seaborn-barplot/31861477"""
    income_bins = customer_income['income'].unique()

    xticks = np.array(np.arange(0, len(income_bins), 5))
    xticklabels = [income_bins[elem] / 1e3 for elem in xticks]
    xticklabels = ["%.1fK" % (elem) for elem in xticklabels]
    xticklabels

    min_income = np.floor(customer_income['income'].min() / 10e3) * 10e3
    max_income = np.ceil(customer_income['income'].max() / 10e3) * 10e3
    xticks = np.linspace(min_income, max_income, 5)

    xticks = np.array(np.arange(0, customer_income.shape[0], 5))

    f, ax = plt.subplots(nrows=4, ncols=1, figsize=(12, 14))

    barobj = sns.barplot(x='income',
                         y='probability',
                         hue='successful',
                         ax=ax[0],
                         data=customer_income)

    barobj.set_xticks(xticks)
    barobj.set_xticklabels(xticklabels, rotation=90)
    barobj.set_xlim((0, len(income_bins)))
    barobj.set_xlabel('Customer Income')
    barobj.set_ylabel('P(Customer\nIncome)')
    barobj.set_title("offerid: %s " % (cur_offer_id))

    barobj = sns.barplot(x='gender',
                         y='percentage',
                         hue='successful',
                         ax=ax[1],
                         data=customer_gender)

    barobj.set_xlabel('Customer Gender')
    barobj.set_ylabel('% of Customers')

    barobj = sns.barplot(x='agerange',
                         y='percentage',
                         hue='successful',
                         ax=ax[2],
                         data=customer_agerange)

    barobj.set_xlabel('Customer Age Range')
    barobj.set_ylabel('% of Customers')

    barobj.set_xticklabels(barobj.get_xticklabels(), rotation=90)

    barobj = sns.barplot(x='membershipstartyear',
                         y='percentage',
                         hue='successful',
                         ax=ax[3],
                         data=membershipstartyear)

    plt.tight_layout()

def explore_customer_offer(offer_number,
                           percent_success,
                           training_data,
                           gender_integer_map):
    """ Builds a customer demographics visualization and computes
    summary statistics for a specific offer
    
    INPUT:
        offer_number: Integer that refers to a specific customer offer
        
        percent_success: DataFrame that describes the success percentage for
                         each offer

        training_data: DataFrame that stores customer demographic data
                       for a specific offer
                       
        gender_integer_map: Dictionary that describes the mapping of a
                            gender string to an integer

    OUTPUT:
        None"""
    cur_offer_id =\
        percent_success.iloc[offer_number]['offerid']

    (customer_income,
     customer_gender,
     customer_agerange,
     membershipstartyear) = select_customer_data(cur_offer_id,
                                                 training_data,
                                                 gender_integer_map)

    print_customer_statistics(cur_offer_id,
                              customer_income,
                              customer_agerange)

    visualize_customer_statistics(cur_offer_id,
                                customer_income,
                                customer_gender,
                                customer_agerange,
                                membershipstartyear)

    print(customer_gender)

    print(membershipstartyear.sort_values('membershipstartyear'))
