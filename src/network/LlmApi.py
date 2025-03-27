import requests
import json
import logging
import time


def post_api_request(model_name, prompt, temp, max_tokens=2048, attempts=3) -> dict | None:
    """ Sends an API request to OpenRouter API to generate completions for the given prompt. """
    if attempts == 0:
        return None
    
    data_json=json.dumps({
            "model": model_name,
            "messages": prompt,
            "max_tokens": max_tokens,
            "temperature": temp,
        })

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer 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",
            "Content-Type": "application/json",
        },
        data=data_json
    )
    
    output = None
    if response.status_code == 200:
        output = response.json()
    else:
        logging.error(f"API request failed with status code {response.status_code}")
        return post_api_request(prompt, temp, max_tokens, attempts-1) if attempts > 0 else None
    if 'error' in output:   # daily API call limits will return status code 200 but with an error messagea
        logging.error(f"API request failed with error: {output['error']}")
        time.sleep(30)
        return post_api_request(prompt, temp, max_tokens, attempts-1) if attempts > 0 else None
    return output