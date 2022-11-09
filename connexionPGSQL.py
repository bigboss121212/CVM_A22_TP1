from array import *
from pickletools import uint8
import psycopg2
import numpy as np

# https://www.psycopg.org/docs/usage.html

class DataBase:
    def __init__(self):    
        self.conn = self.getConnection()

    def getConnection(self):
        try:
            connection = psycopg2.connect("dbname=postgres user=postgres port=5432 password=ASDasd123")
            return connection
        except psycopg2.Error as e:
            print("Unable to connect!", e.pgerror, e.diag.message_detail)
        else:
            print("Connected!")
            
    def closeConnection(self):
        self.conn.close()
        
    def printInfos(self):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM klustr.available_datasets();")
        #print(cur.description)
        #value = cur.fetchone()
        #print(f'one > {value}')
        # data = []
        # for i, emp in enumerate(cur):
        #     data.append(str(emp[0]) + str(emp[1]) + str(emp[2]) + str(emp[3]) + str(emp[4]) + str(emp[5]) + str(emp[6]) + str(emp[7]) + str(emp[8]))
        # print(data)
        self.closeConnection()
    
    def getInfos(self):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM klustr.available_datasets();")
        data = []
        for i, emp in enumerate(cur):
            # title = 
            data.append(str(emp[1]) + " " + "[" + str(emp[5]) + "]" + "[" + str(emp[8]) + "]")
        self.closeConnection()
        return data



        
#
#
# def main():
#     db = DataBase()
#     db.printInfos()
#
# if __name__ == '__main__':
#     main()

        