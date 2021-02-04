import paramiko
from scp import SCPClient
import logging


# Constants
host = "ec2-54-198-36-217.compute-1.amazonaws.com"
port = 22
username = "ec2-user"
password = ""
rsa_key_filename = '/Temp/BigData/Key/bigdataproject.pem'
files_path_list = ['/Temp/BigData/categories_string.csv', '/Temp/BigData/label.csv', '/Temp/BigData/data.json']
aws_directory_path = '/home/ec2-user/'


def main():

    # Create ssh client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())


    # Connect to ssh server
    try:
        key = paramiko.RSAKey.from_private_key_file(rsa_key_filename)
        ssh.connect(hostname=host, port=port, username=username, password=password, pkey=key)
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
    for f in files_path_list:
        put_file(scp, f, aws_directory_path)
        logging.info('File ' + f + ' sent')

    # Close clients
    scp.close()
    ssh.close()


# Put file with scp client on AWS instance
def put_file(scp, host_file_path, aws_file_path):
    scp.put(host_file_path, aws_file_path)

main()
