
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
FILES_PATH_LIST = ['/Temp/BigData/categories_string.csv', '/Temp/BigData/label.csv', '/Temp/BigData/data.json']
AWS_DIRECTORY_PATH = '/home/ec2-user/'


## Export files

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

    # Put file on AWS instance
    for f in FILES_PATH_LIST:
        put_file(scp, f, AWS_DIRECTORY_PATH)
        logging.info('File ' + f + ' sent')

    # Close clients
    scp.close()
    ssh.close()


# Put file with scp client on AWS instance
def put_file(scp, host_file_path, aws_file_path):
    scp.put(host_file_path, aws_file_path)

main()
