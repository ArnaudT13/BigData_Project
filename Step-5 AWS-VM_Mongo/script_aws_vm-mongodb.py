
## Imports

import paramiko
import os
from scp import SCPClient
import logging
import pandas as pd
from pymongo import MongoClient

## Constants

HOST                        = "ec2-34-238-143-69.compute-1.amazonaws.com"
PORT                        = 22
USERNAME                    = "ec2-user"
PASSWORD                    = ""
RSA_KEY_FILENAME            = '/Users/cedri/Documents/Cours Ingenieur/S9/Big Data Project/bigdata.pem'
AWS_FILE_PATH               = '/home/ec2-user/notebooks/Predict_result_SVD.csv'
HOST_WINDOWS_DIRECTORY_PATH = '/Users/cedri/Desktop/temp'

DB_NAME         = "bigdata_db"
COLLECTION_NAME = "predicts"

## Get predict file

def main():

    # Create ssh client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())


    # Connect to ssh server
    try:
        key = paramiko.RSAKey.from_private_key_file(RSA_KEY_FILENAME)
        ssh.connect(hostname=HOST, port=PORT, username=USERNAME, password=PASSWORD, pkey=key)
    except paramiko.ssh_exception.NoValidConnectionsError as e:
        logging.error('Server authentification --> %s', e.args[1])
        return
    except paramiko.ssh_exception.AuthenticationException:
        logging.error('Server authentification --> incorrect login or password')
        return
    except paramiko.ssh_exception.SSHException:
        logging.error('Server connection --> incorrect port')
        return
    except Exception:
        logging.error('Connection error')
        return

    # Create scp client
    scp = SCPClient(ssh.get_transport())
    get_file(scp, AWS_FILE_PATH, HOST_WINDOWS_DIRECTORY_PATH)

    # Close clients
    scp.close()
    ssh.close()

    # Put file to mongodb
    put_to_mongodb(HOST_WINDOWS_DIRECTORY_PATH)


# Put file with scp client on AWS instance
def get_file(scp, aws_file_path, host_directory_path):
    print("[Info] Get files with SCP :", host_directory_path)
    #scp.get(aws_file_path, host_directory_path)
    




def put_to_mongodb(host_directory_path):

    df = pd.read_csv(host_directory_path)
    df.head()

    # MongoClient
    print("[Info] Create MongoDB Client (port : 27017) ")
    client = MongoClient(port=27017)

    print("[Info] Database :", DB_NAME)
    print("[Info] Collection :", COLLECTION_NAME)
    mydb = client[DB_NAME]
    mycol = mydb[COLLECTION_NAME]

    print("[Info] Inserting data")
    for index, row in df.iterrows():
        doc = {"_id": row['Id'],
            "description": row['description'], 
            "gender": row['gender'],
            "prediction": row['Prediction']
            }

        result = mydb.predicts.insert_one(doc)
        printProgressBar (index, len(df), prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r")




def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


main()