from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.functions import udf, unbase64
from pyspark.ml.feature import StopWordsRemover, Tokenizer
import base64
from base64 import  b64decode as decode

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

def get_sentiment_dict():
    sentiment_dict = {}
    with open("./yelp-data/AFINN-111.txt") as f:
        for line in f:
            l_list = line.strip().split('\t')
            sentiment_dict[l_list[0]] = l_list[1]
    return sentiment_dict

def get_stopword_list():
    stop_list = ['']
    with open("./yelp-data/stopwords.txt") as f:
        for line in f.readlines():
            stop_list.append(line.strip())
    return stop_list

def strip_punc(string):
    punc='!"#$%&()*+,-./:;<=>?@[\]^`{|}~'
    for ch in punc:
        string = string.replace(ch, '')
    return string

def sum_sentiment(value):
    afinn = get_sentiment_dict()
    total = 0
    for word in value:
        total += int(afinn.get(word,0))
    return total

#forrige øving
def create_review_df(spark: SparkSession):
    stopwords = get_stopword_list()
    

    df = spark.read.csv(l_file[1],header=True,sep="\t")
    
    #unbase
    df = df.withColumn("review_text", unbase64(df["review_text"]).cast("string"))

    #tokenize
    tokenizer = Tokenizer(inputCol="review_text", outputCol="temp")
    df = tokenizer.transform(df)
    df = df.drop("review_text")

    #stopwords
    remover = StopWordsRemover(inputCol="temp", outputCol="words", stopWords=stopwords)
    df = remover.transform(df)
    df = df.drop("temp")

    #punctuation
    rdd = df.rdd.map(tuple)
    rdd = rdd.map(lambda x: (x[2],[strip_punc(y) for y in x[4]]))

    #sentiment
    rdd = rdd.mapValues(lambda v: sum_sentiment(v))
    sentiment_rdd = rdd.reduceByKey(lambda a,b: a+b)
    sorted_rdd = sentiment_rdd.sortByKey(ascending=False)

    print(sorted_rdd.take(3))
    


def tokenize_review(review):
    tokenized = 2


spark = SparkSession.builder.appName("hello_dataframe").config("spark.some.config.option", "some-value").getOrCreate()
df = create_review_df(spark)

#sentiment_dict = get_sentiment_dict()

