import os
import pickle
import hashlib
from functools import wraps
import logging
import time

def cache_decorator(cache_file_name):
    """ Caches the return value of a decorating function """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create cache directory if it doesn't exist
            cache_dir = 'cache'
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            # Create a unique cache key based on function name and arguments
            cache_key = hashlib.md5((func.__name__ + str(args) + str(kwargs)).encode()).hexdigest()
            cache_path = os.path.join(cache_dir, cache_key + '.pkl')
            cache_path = os.path.join(cache_dir, cache_file_name + cache_key + '.pkl')

            # Check if the result is already cached
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as cache_file:
                    return pickle.load(cache_file)

            # Call the function and cache the result
            result = func(*args, **kwargs)
            with open(cache_path, 'wb') as cache_file:
                pickle.dump(result, cache_file)

            return result
        return wrapper
    return decorator


def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the class name if the function is a method
        class_name = None
        if args and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
        
        # Start timing
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Calculate the execution time
        elapsed_time = end_time - start_time
        
        # Build and log the message
        if class_name:
            logging.info(f"Executed {class_name}.{func.__name__} in {elapsed_time:.4f} seconds")
        else:
            logging.info(f"Executed {func.__name__} in {elapsed_time:.4f} seconds")
        
        return result
    
    return wrapper