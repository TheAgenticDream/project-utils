import json
import pathlib
import os


class Config(object):
    def __init__(self, kubernetes: bool = None, config_file: str = None):
        if os.getenv('k8_mode'):
            self.k8_mode()
        else:
            if not config_file:
                with open(pathlib.Path(__file__).parent / 'encoded.config.json') as files:
                    self.data = json.load(files)
            else:
                with open(config_file) as files:
                    self.data = json.load(files)

    def k8_mode(self):
        with open('/mnt/tmp/config/encoded.config.json') as files:
            file = files.read()
        with open(file) as files:
            self.data = json.load(files)
            files.close()
        return

    def return_config_database(
            self,
            database_encoded: bool = None,
            database_database: bool = None,
            database_user: bool = None,
            database_password: bool = None,
            database_host: bool = None,
            database_port: bool = None,
            database_max_connection: bool = None,
            database_min_connection: bool = None):
        db = self.data['database']
        if database_encoded:
            return db['encoded']
        if database_database:
            return db['database']
        if database_user:
            return db['user']
        if database_password:
            return db['password']
        if database_host:
            return db['host']
        if database_port:
            return db['port']
        if database_max_connection:
            return db['max_connection']
        if database_min_connection:
            return db['min_connection']
        return None

    def return_config_secrets(
            self,
            secrets_encoded: bool = None,
            secrets_fernet_key: bool = None,
            secrets_salt_key: bool = None,
            secrets_conn: bool = None):
        db = self.data['secrets']
        if secrets_encoded:
            return db['encoded']
        if secrets_fernet_key:
            return db['fernet_key']
        if secrets_salt_key:
            return db['salt_key']
        if secrets_conn:
            return db['conn']
        return None

    def return_config_ai_keys(
            self,
            ai_keys_encoded: bool = None,
            ai_keys_OpenAi_key: bool = None,
            ai_keys_OpenRouter_Key: bool = None,
            ai_keys_Ollama_Key: bool = None):
        db = self.data['ai_keys']
        if ai_keys_encoded:
            return db['encoded']
        if ai_keys_OpenAi_key:
            return db['OpenAi_key']
        if ai_keys_OpenRouter_Key:
            return db['OpenRouter_Key']
        if ai_keys_Ollama_Key:
            return db['Ollama_Key']
        return None

    def return_config_orchestration_engine(
            self,
            orchestration_engine_encoded: bool = None,
            orchestration_engine_host: bool = None,
            orchestration_engine_port: bool = None,
            orchestration_engine_api_key: bool = None):
        db = self.data['orchestration_engine']
        if orchestration_engine_encoded:
            return db['encoded']
        if orchestration_engine_host:
            return db['host']
        if orchestration_engine_port:
            return db['port']
        if orchestration_engine_api_key:
            return db['api_key']
        return None

    def return_config_environment(self,
                                  environment_encoded: bool = None,
                                  environment_log_level: bool = None,
                                  environment_environment: bool = None):
        db = self.data['environment']
        if environment_encoded:
            return db['encoded']
        if environment_log_level:
            return db['log_level']
        if environment_environment:
            return db['environment']
        return None

    def return_config_api_server(
            self,
            api_server_encoded: bool = None,
            api_server_host: bool = None,
            api_server_port: bool = None):
        db = self.data['api_server']
        if api_server_encoded:
            return db['encoded']
        if api_server_host:
            return db['host']
        if api_server_port:
            return db['port']
        return None


if __name__ == '__main__':
    print('On Main')
