# Chevy Robertson (crr78@georgetown.edu)
# ANLY 590: Neural Nets & Deep Learning
# 11/11/2021

# This code demonstrates how the Twitter API was accessed to extract tweet data
# with the hashtag, "finance."

# calling/loading the package
library(rtweet)

# API keys and access tokens needed for authorizing data request
API_Key             = "[REDACTED]"
API_Secret_Key      = "[REDACTED]"
Access_Token        = "[REDACTED]"
Access_Token_Secret = "[REDACTED]"

# using the keys and tokens as parameters in create_token() for sending 
# request to generate oauth token for authorizing data access
create_token(consumer_key    = API_Key,
             consumer_secret = API_Secret_Key, 
             access_token    = Access_Token, 
             access_secret   = Access_Token_Secret)

# returning a parsed data frame consisting of text data from a maximum of
# 200 recent and popular tweets (excluding retweets) that include all
# words in the phrase “#finance” and assigning data frame to "finance_tweets"
finance_tweets <- rtweet::search_tweets(q="#finance", n=10000, type="mixed",
                                        include_rts=F, parse=T)

# extract the date and text columns of the dataframe, assign to new dataframe
df <- data.frame(finance_tweets$created_at, finance_tweets$text)

# write the tweets to a csv file
write.csv(df, file="recent_finance_tweets.csv")


