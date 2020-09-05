# installing pyspark
#!pip install pyspark

# importing some libraries
import pandas as pd
import os
import pyspark
from pyspark import SparkContext
sc = SparkContext("local", "Simple App")
from pyspark.sql import SQLContext
# import findspark       #include to run on cluster
# findspark.init()       #include to run on cluster

# stuff we'll need for text processing
from nltk.corpus import stopwords
import nltk
# nltk.download('stopwords')
import re as re
from pyspark.ml.feature import CountVectorizer, IDF
sqlContext = SQLContext(sc)
# stuff we'll need for building the model
from pyspark.mllib.linalg import Vector, Vectors
#from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.ml.clustering import LDA, LDAModel
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.functions import size
from pyspark.sql.functions import udf, col
from pyspark.ml.feature import Tokenizer
from pyspark.sql.functions import udf, struct
import pyspark.sql.types as T
import string
from pyspark.sql import Row
import timeit
# path to the file
#* file path on the server example :"hdfs://master:9000/user/user12/topicModelling/___.csv"
path_to_input_csv_file = "NEWS_Content_100.csv"
start = timeit.default_timer()  #start time
# reading the data
data = sqlContext.read.format("csv") \
    .options(header='true', inferschema='true') \
    .load(os.path.realpath(path_to_input_csv_file))  # remove os.path.realpath() in order to run on cluster

#=================================== PREPROCESSING ==========================================#
start_cleaning = timeit.default_timer()

# selecting contents column
contents = data.rdd.map(lambda x: x['Content']).filter(lambda x: x is not None)
# stop words (useless words such as I,you,he ...)
StopWords = stopwords.words("english")

tokens = contents                                                   \
    .map(lambda document: document.strip().lower())               \
    .map(lambda document: re.split(" ", document))                 \
    .map(lambda word: [x for x in word if x.isalpha()])           \
    .map(lambda word: [x for x in word if len(x) > 3])           \
    .map(lambda word: [x for x in word if x not in StopWords])    \
    .zipWithIndex()
df_txts = sqlContext.createDataFrame(tokens, ["list_of_words", 'index'])
stop_cleaning = start = timeit.default_timer()

# applying TF-IDF
# *search for TF-IDF formatting
TFIDF_start = timeit.default_timer()

# TF
cv = CountVectorizer(inputCol="list_of_words",
                     outputCol="raw_features", vocabSize=5000, minDF=10.0)
cvmodel = cv.fit(df_txts)
result_cv = cvmodel.transform(df_txts)
# IDF
idf = IDF(inputCol="raw_features", outputCol="features")
idfModel = idf.fit(result_cv)
result_tfidf = idfModel.transform(result_cv)
# result_tfidf.show()

TFIDF_stop  = timeit.default_timer()

df_model = result_tfidf.select('index', 'list_of_words', 'features')
# df_model.show()
#========================================================================================#


#==================================CREATING LDA MODEL=======================================#
LDA_start = timeit.default_timer()

# number of Topics
num_topics = 5
max_iterations = 100
lda_model = LDA(k=num_topics, maxIter=max_iterations)
model = lda_model.fit(df_model)

# showing the model
model.describeTopics(5).show()
model.describeTopics().first()
# transforming the model
transformed = model.transform(df_model)
transformed.show()

# Final Result
datadf = data.selectExpr("_c0 as Index", "Content as Content")
datadf.show()
result = datadf.join(transformed, on="Index", how="left")
LDA_stop = timeit.default_timer()

# showing the final result
result.show()
# showing first row of the final result for more understanding
result.first()


sc.stop()
stop = timeit.default_timer()
print('cleaning time:', stop_cleaning - start_cleaning)
print('TF-IDF time:', TFIDF_stop - TFIDF_start)
print('LDA time:', LDA_stop - LDA_start )
print('total time:', stop - start)