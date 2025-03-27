import os
import json
from pathlib import Path

# Default path for config file - in user's home directory
DEFAULT_CONFIG_PATH = os.path.join(Path.home(), ".autodfm_config.json")
TOKENS_JSON_PATH = os.path.join(os.path.dirname(__file__), "tokens.json")

# Default configuration
DEFAULT_CONFIG = {
    "api_tokens": {
        "openai_api_key": "",
        "huggingface_api_token": ""
    },
    "export": {
        "output_dir": "dfm_exports"
    }
}

def load_tokens():
    """Load tokens from tokens.json file"""
    if os.path.exists(TOKENS_JSON_PATH):
        try:
            with open(TOKENS_JSON_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading tokens.json: {e}")
            return {}
    return {}

def load_config(config_path=DEFAULT_CONFIG_PATH):
    """Load configuration from file, or create default if doesn't exist"""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading config: {e}")
            return DEFAULT_CONFIG
    else:
        # Create default config file
        save_config(DEFAULT_CONFIG, config_path)
        return DEFAULT_CONFIG

def save_config(config, config_path=DEFAULT_CONFIG_PATH):
    """Save configuration to file"""
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except IOError as e:
        print(f"Error saving config: {e}")
        return False

def get_api_token(token_name, config_path=DEFAULT_CONFIG_PATH):
    """Get API token from config, falls back to environment variable"""
    # First check tokens.json
    tokens = load_tokens()
    if token_name.upper() in tokens:
        return tokens[token_name.upper()]
    
    # Then check environment
    env_token = os.environ.get(token_name.upper())
    if env_token:
        return env_token
    
    # Finally check config file
    config = load_config(config_path)
    return config.get("api_tokens", {}).get(token_name.lower(), "")

def set_api_token(token_name, token_value, config_path=DEFAULT_CONFIG_PATH):
    """Set API token in config file"""
    config = load_config(config_path)
    if "api_tokens" not in config:
        config["api_tokens"] = {}
    
    config["api_tokens"][token_name.lower()] = token_value
    return save_config(config, config_path)

def get_export_dir(config_path=DEFAULT_CONFIG_PATH):
    """Get export directory from config"""
    config = load_config(config_path)
    return config.get("export", {}).get("output_dir", "dfm_exports")

def set_export_dir(dir_path, config_path=DEFAULT_CONFIG_PATH):
    """Set export directory in config"""
    config = load_config(config_path)
    if "export" not in config:
        config["export"] = {}
    
    config["export"]["output_dir"] = dir_path
    return save_config(config, config_path) 