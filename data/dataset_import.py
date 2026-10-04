import kagglehub

# Download latest version
path = kagglehub.dataset_download("ashoknepal/nepali-fake-news-detection")

print("Path to dataset files:", path)