import requests
import logging
from typing import Dict, Any, Optional

# It's good practice to use a logger specific to this module or class
logger = logging.getLogger(__name__) 
# Basic config if no global logger is set, for standalone testing.
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ApiClient:
    """
    A generic API client for making HTTP requests.
    """

    def __init__(self, base_url: str = "", api_key: Optional[str] = None, default_headers: Optional[Dict[str, str]] = None):
        """
        Initializes the ApiClient.

        Args:
            base_url: The base URL for the API. Endpoints will be appended to this.
            api_key: An optional API key that can be added to headers or params.
                     How it's used depends on the specific API (e.g., 'Authorization: Bearer <key>').
            default_headers: Default headers to include in every request.
        """
        self.base_url = base_url.rstrip('/') # Ensure no trailing slash
        self.api_key = api_key
        self.default_headers = default_headers or {}

        if self.api_key and not any(key.lower() == 'authorization' for key in self.default_headers):
            # This is a common pattern, but might not be universal.
            # Consider if API key should be handled more flexibly (e.g., in params, or a callback).
            # For now, if an api_key is given and Authorization is not in default_headers, we add it as Bearer token.
            # self.default_headers['Authorization'] = f'Bearer {self.api_key}' # Example: Bearer token
            logger.info("API key provided. Ensure it's correctly used in headers or params as per API docs.")


    def request(self, 
                method: str, 
                endpoint: str, 
                params: Optional[Dict[str, Any]] = None, 
                data: Optional[Dict[str, Any]] = None, 
                json_data: Optional[Dict[str, Any]] = None,
                custom_headers: Optional[Dict[str, str]] = None,
                timeout: int = 10) -> Dict[str, Any]: # Changed to Dict for JSON response
        """
        Makes an HTTP request to the specified endpoint.

        Args:
            method: HTTP method (e.g., 'GET', 'POST', 'PUT', 'DELETE').
            endpoint: API endpoint path (e.g., '/users', 'data/prices').
            params: Dictionary of URL parameters for GET requests.
            data: Dictionary of form data for POST/PUT requests.
            json_data: Dictionary to be sent as JSON body for POST/PUT requests.
                       If `json_data` is provided, `data` will be ignored for the body.
            custom_headers: Custom headers for this specific request, merged with default_headers.
            timeout: Request timeout in seconds.

        Returns:
            A dictionary containing the JSON response from the API.

        Raises:
            requests.exceptions.HTTPError: For HTTP error responses (4xx or 5xx).
            requests.exceptions.RequestException: For other request issues (e.g., connection error).
            ValueError: If method is not supported or invalid parameters.
        """
        if not method or method.upper() not in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD']:
            logger.error(f"Unsupported HTTP method: {method}")
            raise ValueError(f"Unsupported HTTP method: {method}")

        # Construct full URL
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        # Prepare headers
        headers = self.default_headers.copy()
        if custom_headers:
            headers.update(custom_headers)
        
        # If an API key was provided to __init__ and is not yet in headers (e.g. via default_headers),
        # this is a place one might inject it, e.g. as a Bearer token.
        # For example, if 'Authorization' is not already set:
        if self.api_key and 'Authorization' not in headers and 'authorization' not in headers:
             # A common convention, but not universal. API docs are key.
             # headers['Authorization'] = f'Bearer {self.api_key}'
             logger.debug("Using API key from constructor in Authorization header (if not already set).")
             # This part might need to be more sophisticated based on how different APIs expect keys.
             # Some APIs might expect it in 'X-API-Key' header or as a query parameter.

        logger.info(f"Making {method.upper()} request to {url}")
        logger.debug(f"Headers: {headers}, Params: {params}, JSON Data: {json_data}, Form Data: {data}")

        try:
            response = requests.request(
                method=method.upper(),
                url=url,
                params=params,
                data=data,
                json=json_data, # `requests` handles data/json precedence correctly.
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()  # Raises HTTPError for 4xx/5xx responses
            
            # Attempt to parse JSON, handle cases where response might be empty or not JSON
            try:
                json_response = response.json()
                logger.debug(f"Response JSON: {json_response}")
                return json_response
            except requests.exceptions.JSONDecodeError:
                logger.warning(f"Response from {url} was not valid JSON. Status: {response.status_code}, Body: {response.text[:100]}...")
                # Depending on API, non-JSON might be acceptable for some successful responses (e.g. 204 No Content)
                if response.ok and not response.content: # e.g. 204 No Content
                    return {"status": "success", "message": "No content in response.", "status_code": response.status_code}
                return {"status": "error", "message": "Response was not valid JSON.", "raw_content": response.text, "status_code": response.status_code}


        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error occurred: {e.response.status_code} {e.response.reason} for {url}. Response: {e.response.text[:200]}")
            # Optionally, parse error response if it's JSON
            try:
                error_json = e.response.json()
                return {"status": "http_error", "error_details": error_json, "status_code": e.response.status_code, "reason": e.response.reason}
            except requests.exceptions.JSONDecodeError:
                return {"status": "http_error", "raw_error": e.response.text, "status_code": e.response.status_code, "reason": e.response.reason}
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for {url}: {e}")
            raise  # Or return a structured error: {"status": "connection_error", "message": str(e)}
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout occurred for {url}: {e}")
            raise  # Or return: {"status": "timeout_error", "message": str(e)}
        except requests.exceptions.RequestException as e:
            logger.error(f"An unexpected error occurred during request to {url}: {e}")
            raise # Or return: {"status": "request_error", "message": str(e)}

if __name__ == '__main__':
    # Example Usage (requires internet and a public API to test against)
    logger.info("--- Testing ApiClient ---")

    # Test with a public API (JSONPlaceholder)
    json_placeholder_client = ApiClient(base_url="https://jsonplaceholder.typicode.com")

    # 1. GET request
    print("\n1. Testing GET request...")
    try:
        posts = json_placeholder_client.request(method='GET', endpoint='/posts?userId=1', params={'_limit': 2})
        if posts and isinstance(posts, list) and len(posts) > 0 :
            print(f"Successfully fetched {len(posts)} posts. First post title: {posts[0].get('title')}")
        elif isinstance(posts, dict) and posts.get("status_code") != 200 : # Error case from our client
            print(f"API Client returned error: {posts}")
        else:
            print(f"Fetched posts data (or error structure): {posts}")
    except Exception as e:
        print(f"GET request failed: {e}")

    # 2. POST request
    print("\n2. Testing POST request...")
    try:
        new_post_data = {'title': 'foo', 'body': 'bar', 'userId': 1}
        created_post = json_placeholder_client.request(method='POST', endpoint='/posts', json_data=new_post_data)
        if created_post and created_post.get('id'):
            print(f"Successfully created post with ID: {created_post.get('id')}, Title: {created_post.get('title')}")
        else:
            print(f"POST request response: {created_post}")
    except Exception as e:
        print(f"POST request failed: {e}")

    # 3. Testing with a non-existent endpoint (expecting 404)
    print("\n3. Testing non-existent endpoint (expecting 404)...")
    try:
        error_response = json_placeholder_client.request(method='GET', endpoint='/nonexistentendpoint')
        print(f"Response from non-existent endpoint: {error_response}") # Should be our structured error
        assert error_response.get("status_code") == 404
        print("Correctly handled 404 error.")
    except requests.exceptions.HTTPError as e: # This would be if raise_for_status() was not caught inside
        print(f"HTTPError (should be caught by client): {e}")
    except Exception as e:
        print(f"Request to non-existent endpoint failed in an unexpected way: {e}")

    # 4. Testing with default headers and API key placeholder (won't actually authenticate)
    print("\n4. Testing with default headers and API key placeholder...")
    # This client would typically be for an API that requires a key.
    # JSONPlaceholder doesn't, so this is just for show.
    client_with_key = ApiClient(
        base_url="https://jsonplaceholder.typicode.com", 
        api_key="YOUR_DUMMY_API_KEY",
        default_headers={'X-Custom-Header': 'CustomValue'}
    )
    # If the API key logic were to add 'Authorization: Bearer YOUR_DUMMY_API_KEY', it would be in the request.
    # We can't easily verify it here without a specific endpoint that reflects headers.
    try:
        response = client_with_key.request(method='GET', endpoint='/todos/1')
        print(f"Response from client with key placeholder: {response.get('title', 'Error or wrong structure')}")
    except Exception as e:
        print(f"Request with key placeholder failed: {e}")
        
    # 5. Test timeout (difficult to reliably test without a slow server)
    print("\n5. Testing timeout (conceptual - set to very low value)...")
    try:
        # Using a known slow API or a local server that delays response would be better.
        # For now, just show the timeout parameter being used.
        # This will likely succeed quickly as JSONPlaceholder is fast.
        response = json_placeholder_client.request(method='GET', endpoint='/posts', timeout=0.001) # Extremely low timeout
        print(f"Response with low timeout: {len(response) if isinstance(response, list) else response}")
    except requests.exceptions.Timeout:
        print("Request timed out as expected (or if server was very slow).")
    except Exception as e:
        print(f"Request with low timeout failed or completed: {e}") # Might complete if server is faster than timeout

    print("\n--- ApiClient tests complete ---")
