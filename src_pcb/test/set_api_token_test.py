#!/usr/bin/env python
import sys
import os

# Import config from parent directory
from .. import config

# Set a test token (this would normally be a real token)
test_openai_token = "test_sk_1234567890abcdefghijklmnopqrstuvwxyz"
test_huggingface_token = "test_hf_1234567890abcdefghijklmnopqrstuvwxyz"

# Set the tokens in config
print(f"Setting test OpenAI API key in config...")
config.set_api_token("openai_api_key", test_openai_token)

print(f"Setting test Hugging Face API token in config...")
config.set_api_token("huggingface_api_token", test_huggingface_token)

# Print current configuration with masked tokens
current_config = config.load_config()

openai_key = current_config.get("api_tokens", {}).get("openai_api_key", "")
huggingface_token = current_config.get("api_tokens", {}).get("huggingface_api_token", "")
export_dir = current_config.get("export", {}).get("output_dir", "dfm_exports")

# Mask tokens for display
if openai_key:
    masked_openai = openai_key[:4] + "*" * (len(openai_key) - 8) + openai_key[-4:]
else:
    masked_openai = "Not set"
    
if huggingface_token:
    masked_huggingface = huggingface_token[:4] + "*" * (len(huggingface_token) - 8) + huggingface_token[-4:]
else:
    masked_huggingface = "Not set"

print("\nCurrent Configuration:")
print(f"OpenAI API Key: {masked_openai}")
print(f"Hugging Face API Token: {masked_huggingface}")
print(f"Export Directory: {export_dir}")
print(f"\nConfig file location: {config.DEFAULT_CONFIG_PATH}")

print("\nNote: These are TEST tokens and won't work with actual API services.")
print("Use this as a template for setting real tokens in a secure manner.") 