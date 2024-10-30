
class APIRequestError(Exception):
    """
    Custom error class for API request failures.

    Parameters
    ----------
    message : str
        Error message to be displayed.
    status_code : int
        HTTP status code of the failed request.
    endpoint : str
        API endpoint that was requested.
    response_text : str, optional
        The response text or message returned from the API.
    """
    def __init__(self, message: str, status_code: int = None, endpoint: str = None, response_text: str = None):
        self.status_code = status_code
        self.endpoint = endpoint
        self.response_text = response_text
        super().__init__(message)
    
    def __str__(self):
        return (f"APIRequestError: {self.args[0]} | "
                f"Status Code: {self.status_code}, "
                f"Endpoint: {self.endpoint}, "
                f"Response: {self.response_text}")
