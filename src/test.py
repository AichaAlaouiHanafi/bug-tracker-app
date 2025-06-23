from transformers import pipeline

# This will try to download a model (and require outbound internet access)
classifier = pipeline("sentiment-analysis")
print(classifier("I love using Hugging Face!"))
