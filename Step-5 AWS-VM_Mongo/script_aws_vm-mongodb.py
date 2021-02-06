
## Imports

import paramiko
from scp import SCPClient
import logging


## Constants

HOST = "ec2-54-91-166-72.compute-1.amazonaws.com"
PORT = 22
USERNAME = "ec2-user"
PASSWORD = ""
RSA_KEY_FILENAME = '/Temp/BigData/Key/bigdataproject.pem'
AWS_FILE_PATH = '/home/ec2-user/MyNotebooks/Notebook_ML_NN.ipynb'
HOST_WINDOWS_DIRECTORY_PATH = '.'


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


# Put file with scp client on AWS instance
def get_file(scp, aws_file_path, host_directory_path):
    scp.get(aws_file_path, host_directory_path)

main()
