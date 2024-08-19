import vertexai
from vertexai.preview.generative_models import GenerativeModel

PROJECT_ID = "poised-renderer-433000-s0"
REGION = "us-central1"

vertexai.init(project=PROJECT_ID, location=REGION)

model = GenerativeModel('gemini-pro')

response = model.generate_content('Count from 1 to 10')