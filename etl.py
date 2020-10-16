import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['KEYS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['KEYS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    # get filepath to song data file
    song_data = input_data+'song_data/*/*/*/*.json'
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    df.createOrReplaceTempView("song_data_table")
    
    songs_table = spark.sql("""
                            SELECT DISTINCT T.song_id as song_id, 
                            T.title as title,
                            T.artist_id as artist_id,
                            T.year as year,
                            T.duration as duration
                            FROM song_data_table T
                            WHERE T.song_id IS NOT NULL
                        """)
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.parquet(os.path.join(output_data, "songs/") , mode="overwrite", partitionBy=["year","artist_id"])

    
    # extract columns to create artists table
    df.createOrReplaceTempView("artist_data_table")
    
    artists_table = spark.sql("""
                            SELECT DISTINCT T.artist_id as artist_id, 
                            T.artist_name as name,
                            T.artist_location as location,
                            T.artist_lattitude as lattitude,
                            T.artist_longitude as longitude
                            FROM song_data_table T
                            WHERE T.artist_id IS NOT NULL
                        """)
    
    # write artists table to parquet files
    artists_table.write.parquet(os.path.join(output_data, "artist/") , mode="overwrite")


def process_log_data(spark, input_data, output_data):
    # get filepath to log data file
    log_data =input_data+'log_data/*/*/*/*.json'

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.filter(df.page == 'NextSong')

    # extract columns for users table    
    df.createOrReplaceTempView("log_data_table")
    
    users_table = spark.sql("""
                            SELECT DISTINCT T.userId as user_id, 
                            T.firstName as first_name,
                            T.lastName as last_name,
                            T.gender as gender,
                            T.level as level
                            FROM log_data_table T
                            WHERE T.userId IS NOT NULL
                        """)
    
    # write users table to parquet files
    users_table.write.parquet(os.path.join(output_data, "users/") , mode="overwrite")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: x/1000, IntegerType())
    df = df.withColumn('start_time', get_timestamp('ts'))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: from_unixtime(x), TimestampType())
    df = df.withColumn('datetime', from_unixtime('start_time'))
    
    # extract columns to create time table
    time_table = df.withColumn("hour",hour("start_time"))\
                    .withColumn("day",dayofmonth("start_time"))\
                    .withColumn("week",weekofyear("start_time"))\
                    .withColumn("month",month("start_time"))\
                    .withColumn("year",year("start_time"))\
                    .withColumn("weekday",dayofweek("start_time"))\
                    .select("ts","start_time","hour", "day", "week", "month", "year", "weekday").drop_duplicates()
    
    # write time table to parquet files partitioned by year and month
    time_table.parquet(os.path.join(output_data, "time/"), mode='overwrite', partitionBy=["year","month"])

    # read in song data to use for songplays table
    song_df = spark.read\
                .format("parquet")\
                .option("basePath", os.path.join(output_data, "songs/"))\
                .load(os.path.join(output_data, "songs/*/*/"))

    songplay_id, start_time, user_id, level, song_id, artist_id, session_id, location, user_agent
    
    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = df.join(song_df, (df.song == song_df.title) & (df.artist == song_df.artist_name) & (df.length == song_df.duration), 'left_outer')\
            .withColumn('songplay_id', monotonically_increasing_id())
            .select(
            df.timestamp,
            col("userId").alias('user_id'),
            df.level,
            song_df.song_id,
            song_df.artist_id,
            col("sessionId").alias("session_id"),
            df.location,
            col("useragent").alias("user_agent"),
            year('datetime').alias('year'),
            month('datetime').alias('month') 
            )
        
    # write songplays table to parquet files partitioned by year and month    
    songplays_table.parquet(os.path.join(output_data, "songplays/"), mode='overwrite', partitionBy=["year","month"])

def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "arn:aws:s3:::datalakeproject222"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
