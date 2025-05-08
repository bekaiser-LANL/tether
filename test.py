import os
import certifi
from anthropic import Anthropic

# Fix SSL cert path for httpx/httpcore
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

os.environ["SSL_CERT_FILE"] = os.path.expanduser("~/cacert.pem")
os.environ["REQUESTS_CA_BUNDLE"] = os.path.expanduser("~/cacert.pem")

client = Anthropic()

response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1000,
    system="You are a scientist.",
    messages=[{"role": "user", "content": "What is quantum entanglement?"}]
)

print(response.content)
