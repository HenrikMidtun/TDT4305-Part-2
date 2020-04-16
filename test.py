from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.functions import explode, udf, unbase64
from pyspark.ml.feature import StopWordsRemover, Tokenizer
import re

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

def top_k_businesses(spark: SparkSession, k: int = 3):
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
    df = df.select(df.business_id, explode(df.words))

    #punctuation
    commaRep = udf(lambda x: re.sub(',$|^.','', x))
    df = df.withColumn("col", commaRep(df["col"]))

    #sentiment
    afinn = get_sentiment_dict()
    sentimentRep = udf(lambda x: afinn.get(x,0))
    df = df.withColumn("col", sentimentRep(df["col"]))

    #summing on business
    df = df.groupBy("business_id")
    df.agg({'col':'sum'}).orderBy("sum(col)",ascending=False).show(k)

spark = SparkSession.builder.appName("hello_dataframe").config("spark.some.config.option", "some-value").getOrCreate()
df = top_k_businesses(spark)


