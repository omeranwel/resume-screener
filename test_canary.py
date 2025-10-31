import requests

# List of golden-set queries (URLs you expect to be successful)
queries = [
    "http://127.0.0.1:8001/screen_resume/",  # Example endpoint (adjust as per your API)
    # Add more test URLs here...
]


def test_canary():
    for url in queries:
        # Send a POST request with all required fields (name, resume, job_description)
        response = requests.post(
            url,
            json={
                "name": "John Doe",
                "resume": "example",
                "job_description": "example",
            },
        )
        # Check if the response status code is 200 (successful)
        assert (
            response.status_code == 200
        ), f"Failed for {url} with status {response.status_code}"


# Run the test
test_canary()
