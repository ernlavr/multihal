import requests
import json
import logging
import time


def post_api_request(model_name, prompt, temp, args, max_tokens=2048, attempts=3) -> dict | None:
    """ Sends an API request to OpenRouter API to generate completions for the given prompt. """
    if attempts == 0:
        return None
    
    data_json=json.dumps({
            "model": model_name,
            "messages": prompt,
            "max_tokens": max_tokens,
            "temperature": temp,
        })

    response = None
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {args.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            data=data_json
        )
    except Exception as e:
        logging.error(f"API request failed with error: {e}")
        return None
    
    output = None
    if response.status_code == 200:
        output = response.json()
    else:
        logging.error(f"API request failed with status code {response.status_code}")
        return None
    if 'error' in output:   # daily API call limits will return status code 200 but with an error messagea
        logging.error(f"API request failed with error: {output['error']}")
        return None
    return output