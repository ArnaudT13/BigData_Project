# pip install scp
# pip install paramiko

import paramiko
from scp import SCPClient
import logging

host = "localhost"

port = 2222
username = "maria_dev"
password = "maria_dev"

VM_DIR = "/home/maria_dev/data/"
files = ['data.json', 'label.csv', 'categories_string.csv']

def main():

    logging.info("Login")

    # Create ssh client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connect to ssh server
    try:
        ssh.connect(host, port, username, password)
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

    # Create scp client
    scp = SCPClient(ssh.get_transport())

    for f in files:

        logging.info("Transfert Start :", f)

        # Get file
        get_file(scp, VM_DIR + f, f)

    # Close clients
    scp.close()
    ssh.close()

    logging.info("Transfert Done")


# Get file with scp client
def get_file(scp, server_file_path, host_file_path):
    scp.get(server_file_path, host_file_path)

main()