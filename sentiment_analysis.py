from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.functions import explode, udf, unbase64
from pyspark.ml.feature import StopWordsRemover, Tokenizer
import re
import string

l_file = [
    "./yelp-data/yelp_businesses.csv"
    ,"./yelp-data/yelp_top_reviewers_with_reviews.csv"
    ,"./yelp-data/yelp_top_users_friendship_graph.csv"
    ]

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

def top_k_businesses(spark: SparkSession, k: int = 3, ascending: bool = False):
    df = spark.read.csv(l_file[1],header=True,sep="\t")
    
    #unbase
    df = df.withColumn("review_text", unbase64(df["review_text"]).cast("string"))

    #tokenize
    tokenizer = Tokenizer(inputCol="review_text", outputCol="temp")
    df = tokenizer.transform(df)
    df = df.drop("review_text")

    #stopwords
    stopwords = get_stopword_list()
    remover = StopWordsRemover(inputCol="temp", outputCol="words", stopWords=stopwords)
    df = remover.transform(df)
    df = df.drop("temp")
    df = df.select(df.business_id, explode(df.words))

    #punctuation
    commaRep = udf(lambda x: x.translate(str.maketrans('','', string.punctuation)))
    df = df.withColumn("col", commaRep(df["col"]))

    #sentiment
    afinn = get_sentiment_dict()
    sentimentRep = udf(lambda x: afinn.get(x,0))
    df = df.withColumn("col", sentimentRep(df["col"]))

    #summing on business
    df = df.groupBy("business_id")
    result = df.agg({'col':'sum'}).orderBy("sum(col)",ascending=ascending).take(k)
    return result


spark = SparkSession.builder.appName("hello_dataframe").config("spark.some.config.option", "some-value").getOrCreate()
''' We had problems with running this on one computer, so this is a fix for that.
locale = spark._jvm.java.util.Locale
locale.setDefault(locale.forLanguageTag("en-US"))
'''
top_k = top_k_businesses(spark,10)

print(top_k)

with open('Result.txt','w') as f:
    for item in top_k:
        f.write("%s\n" %str(item))
