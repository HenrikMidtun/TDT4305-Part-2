from pyspark import SparkContext


l_file = [
    "./yelp-data/yelp_businesses.csv"
    ,"./yelp-data/yelp_top_reviewers_with_reviews.csv"
    ,"./yelp-data/yelp_top_users_friendship_graph.csv"
    ]

def load_data(context,url):
    data = context.textFile(url)
    header = data.first()
    data = data.filter(lambda line: line != header)
    return data

with open("./yelp-data/AFINN-111.txt") as f:
    sentiment_dict = {}
    for line in f:
        l_list = line.strip().split('\t')
        sentiment_dict[l_list[0]] = l_list[1]
    print(sentiment_dict)

