from abc import ABC, abstractmethod
import psycopg2 as pg


class KlustRDAO(ABC):
    def __init__(self):
        self._translated = True
        self._rotated = True
        self._scaled = True
        self._exclusive = False

    @property
    def translated(self):
        return self._translated

    @property
    def rotated(self):
        return self._rotated

    @property
    def scaled(self):
        return self._scaled

    @property
    def exclusive(self):
        return self._exclusive

    def set_transformation_filters(self, translated=True, rotated=True, scaled=True, exclusive=True):
        self._translated = translated
        self._rotated = rotated
        self._scaled = scaled
        self._exclusive = exclusive

    @property
    @abstractmethod
    def is_available(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def total_label_image_count(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def available_datasets(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def available_labels(self):
        raise NotImplementedError

    @abstractmethod
    def labels_from_dataset(self, dataset_name):
        raise NotImplementedError

    @abstractmethod
    def image_from_label(self, label_id):
        raise NotImplementedError

    @abstractmethod
    def image_from_dataset_label(self, dataset_name, label_id, training_image):
        raise NotImplementedError

    @abstractmethod
    def image_from_dataset(self, dataset_name, training_image):
        raise NotImplementedError


# TO DO : APPLY TRANSFORMATION FILTERS!!! AND move to serverside function
class PostgreSQLKlustRDAO(KlustRDAO):
    def __init__(self, pg_connection_credential, quit_if_connection_failed=False):
        self._pg_connection_credential = pg_connection_credential
        try:
            self.pg_connection = pg.connect(
                self._pg_connection_credential.connection_string)
            self.pg_cursor = self.pg_connection.cursor()
            self._is_available = True
        except Exception as error:
            self._is_available = False
            print(
                'La connection à la base de données a échouée avec le message suivant :')
            print('-' * 80)
            print(type(error))
            print(error)
            print('-' * 80)
            if quit_if_connection_failed:
                quit()

    def _execute_simple_query(self, query, param_to_bind=tuple()):
        if self.is_available:
            try:
                self.pg_cursor.execute(query, param_to_bind)
                return self.pg_cursor.fetchall()
            except Exception as error:
                print(
                    'PostgreSQLKlustRDAO : erreur de la requete avec le message suivant :')
                print('-' * 80)
                print(type(error))
                print(error)
                print(f'Avec la requete :\n{query}')
                print('-' * 80)
        else:
            print('PostgreSQLKlustRDAO n\'est pas disponible.')
        return None

    @property
    def is_available(self):
        return self._is_available and not self.pg_connection.closed

    @property
    def total_label_image_count(self):
        query = 'SELECT * FROM klustr.label_image_total_count();'
        return self._execute_simple_query(query)

    @property
    def available_datasets(self):
        return self._execute_simple_query('''SELECT * FROM klustr.data_set_info;''')

    @property
    def available_labels(self):
        return self._execute_simple_query('''SELECT * FROM klustr.available_labels;''')

    def labels_from_dataset(self, dataset_name):
        return self._execute_simple_query(f'''SELECT * FROM klustr.select_label_from_data_set(%s);''', (dataset_name,))

    def image_from_label(self, label_id):
        return self._execute_simple_query(
            f'''SELECT 	* FROM klustr.select_image_by_label_and_transformation(%s, %s, %s, %s, %s);''',
            (label_id, self._translated, self._rotated, self._scaled, self._exclusive))

    def image_from_dataset_label(self, dataset_name, label_id, training_image):
        return self._execute_simple_query(
            f'''SELECT * FROM klustr.select_image_from_data_set(%s, %s, %s);''',
            (dataset_name, label_id, training_image))

    def image_from_dataset(self, dataset_name, training_image):
        return self._execute_simple_query(
            f'''SELECT * FROM klustr.select_image_from_data_set(%s, %s);''',
            (dataset_name, training_image))

    def total_label_name(self):
        query = 'SELECT * FROM klustr.label;'
        return self._execute_simple_query(query)

    def get_dataset_tests(self, nom):
        return self._execute_simple_query(
            f'''SELECT * FROM klustr.select_image_from_data_set(%s, %s);''',
            (nom, True))

    def get_image_test(self, nom):
        return self._execute_simple_query(
            f'''select img_data from klustr.image where name=(%s);''',
            (nom,))

    def get_label_test(self, nom):
        return self._execute_simple_query(
            f'''select name from klustr.label where id =(%s);''',
            (nom,))

    def get_current_dataset_info(self, name):
        return self._execute_simple_query(
            f'''SELECT * FROM klustr.available_datasets() WHERE name =(%s);''',
            (name,))
