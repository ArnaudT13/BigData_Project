echo "SCRIPT HDFS -> HOST"
mkdir /home/maria_dev/data
hdfs dfs -get BigData/* /home/maria_dev/data
echo "DONE"