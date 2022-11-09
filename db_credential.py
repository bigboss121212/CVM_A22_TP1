from abc import ABC, abstractmethod

class DBCredential(ABC):

    def __init__(self, host, port, database, user, password):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password

    @property
    @abstractmethod
    def connection_string(self):
        raise NotImplementedError


class PostgreSQLCredential(DBCredential):

    def __init__(self, password='', host='localhost', port=5432, database='postgres', user='postgres'):
        super().__init__(host, port, database, user, password)

    @property
    def connection_string(self):
        return f"host='{self.host}' port={self.port} dbname='{self.database}' user='{self.user}' password='{self.password}'"