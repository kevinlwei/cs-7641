import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split

#Final values:
#counties = Pandas dataframe containing county attributes
#dem = Pandas series containing winners of the Democratic primary by county (-1 = Hillary, 1 = Bernie)
#tweets = Pandas dataframe containing tweet attributes
#bot = Pandas series containing whether the tweet was from a bot or not (-1 = no, 1 = yes)

#import 2016 election data
#imports raw county data
countyheaders = ['fips','area_name','state_abbreviation','pop2014','pop2010est','popchange','pop2010','popunder5','popunder18','popover65','popf','popwhite','popblack','popnative','popasian','popnhpi','popmixed','pophispanic','popnonhispanicwhite','popsamehousehold','popforeignborn','popnonnativeenglish','popgradHS','popgradBachelors','veteranscount','meancommutetime','housingunits','homeownershiprate','housingunits-multiunitspercent','housingunits-medianvalue','households','personsperhousehold','percapitaincome','householdmedianincome','belowpovertypercent','privatenonfarmscount','privatenonfarmemployments','privatenonfarmemploymentschange','nonemployercount','firmcount','firmsblackcount','firmsnativecount','firmsasiancount','firmsnhpicount','firmshispaniccount','firmswomencount','manufacturersshipments','merchantwholesalersales','retailsales','retailsalespercapita','hotelandfoodsales','buildingpermitscount','squaremiles','populationsquaremiles']
#raw_county_data = py.read_csv('2016-us-election/county_facts.csv')
raw_county_data = pd.read_csv(filepath_or_buffer='2016-us-election/county_facts.csv', header=0, names=countyheaders)

#removes rows with blank state abbreviations
raw_county_data.dropna(subset=['state_abbreviation'], inplace=True)

#remove one row where there is a tie
raw_county_data = raw_county_data[raw_county_data['fips'] != '56045'] #we randomly assign values below, but we remove this row as an outlier (there were 2 Democratic votes total in this county)

#imports raw 2016 results data
raw_county_winners = pd.read_csv(filepath_or_buffer='2016-us-election/primary_results.csv', header=0)

dem_winners = pd.DataFrame()

fips = raw_county_data['fips'].tolist()

for i in fips:
    bernie_county = raw_county_winners[(raw_county_winners['party'] == 'Democrat') & (raw_county_winners['candidate'] == 'Bernie Sanders') & (raw_county_winners['fips'] == i)]
    hillary_county = raw_county_winners[(raw_county_winners['party'] == 'Democrat') & (raw_county_winners['candidate'] == 'Hillary Clinton') & (raw_county_winners['fips'] == i)]

    if not (hillary_county.empty or bernie_county.empty):
        bernie_votes = bernie_county['votes']
        bernie_votenum = bernie_votes.iloc[0]
        hillary_votes = hillary_county['votes']
        hillary_votenum = hillary_votes.iloc[0]

        if(bernie_votenum > hillary_votenum):
            dem_winners = dem_winners.append(bernie_county)
        elif(bernie_votenum < hillary_votenum):
            dem_winners = dem_winners.append(hillary_county)
        else:
            random.seed(1)
            winner = random.choice([0,1])
            if(winner == 0):
                dem_winners = dem_winners.append(bernie_county)
            else:
                dem_winners = dem_winners.append(hillary_county)
    elif bernie_county.empty:
        dem_winners = dem_winners.append(hillary_county)
    elif hillary_county.empty:
        dem_winners = dem_winners.append(bernie_county)

dem_winners.columns = ['state', 'state_abbreviation', 'county', 'fips', 'party', 'candidate', 'votes', 'fraction_votes']

#final dataframe containing county attributes
raw_county_data = raw_county_data[raw_county_data['fips'].isin(dem_winners['fips'])]
counties = raw_county_data[['fips','state_abbreviation','pop2014','pop2010est','popchange','pop2010','popunder5','popunder18','popover65','popf','popwhite','popblack','popnative','popasian','popnhpi','popmixed','pophispanic','popnonhispanicwhite','popsamehousehold','popforeignborn','popnonnativeenglish','popgradHS','popgradBachelors','veteranscount','meancommutetime','housingunits','homeownershiprate','housingunits-multiunitspercent','housingunits-medianvalue','households','personsperhousehold','percapitaincome','householdmedianincome','belowpovertypercent','privatenonfarmscount','privatenonfarmemployments','privatenonfarmemploymentschange','nonemployercount','firmcount','firmsblackcount','firmsnativecount','firmsasiancount','firmsnhpicount','firmshispaniccount','firmswomencount','manufacturersshipments','merchantwholesalersales','retailsales','retailsalespercapita','hotelandfoodsales','buildingpermitscount','squaremiles','populationsquaremiles']].copy()

#one-hot encoding of state abbreviations
counties = pd.get_dummies(counties,prefix=['state_abbreviation'])

counties = counties.sort_values(by=['fips'])
counties.drop(columns='fips',inplace=True)

#final dataframe containing dem winners by county
dem = dem_winners[['fips','candidate']].copy()
dem = dem.sort_values(by=['fips'])

replacement_dict = {'Hillary Clinton':-1,'Bernie Sanders':1}

dem.replace(to_replace=replacement_dict,inplace=True)
dem.drop(columns='fips',inplace=True)

dem = dem.squeeze()

#import twitter dataset
raw_tweet_data = pd.read_csv(filepath_or_buffer='combo5050.csv', header=0)

tweets = raw_tweet_data.sample(n=3000,random_state=1)
#tweets = tweets[['follower_count','following_count','follower-to-following_ratio','hashtags','links','upper','char_count','tweet_word_count','average_num_of_words','bot_or_not']].copy()
tweets = tweets[['hashtags','links','upper','char_count','tweet_word_count','average_num_of_words','bot_or_not']].copy()
tweets.replace([np.inf, -np.inf], np.nan, inplace=True)
tweets.dropna(axis=0,how='any',inplace=True)

bot = tweets[['bot_or_not']].copy()
bot.replace(0,-1,inplace=True)
bot = bot.squeeze()

tweets.drop(columns='bot_or_not',inplace=True)
#print(tweets.head)
#print(tweets.tail)
#print(tweets.columns)

x_train_2016, x_test_2016, y_train_2016, y_test_2016, train_2016_indices, test_2016_indices = train_test_split(counties,dem,list(range(0,len(counties.index))),train_size=0.70,random_state=1)
x_train_tweets, x_test_tweets, y_train_tweets, y_test_tweets, train_tweets_indices, test_tweets_indices = train_test_split(tweets,bot,list(range(0,len(tweets.index))),train_size=0.70,random_state=1)

#print(x_train_2016)
#print(x_test_2016)
#print(y_train_2016)
#print(y_test_2016)
#print(x_train_tweets)
#print(x_test_tweets)
#print(y_train_tweets)
#print(y_test_tweets)
