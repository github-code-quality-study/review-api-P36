import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'

# Load reviews from CSV file
reviews = pd.read_csv('data/reviews.csv').to_dict('records')

# Ensure all reviews have sentiment scores
for review in reviews:
    if 'ReviewBody' in review:
        review['sentiment'] = sia.polarity_scores(review['ReviewBody'])

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def filter_reviews(self, reviews, query_params):
        # Filter by location
        if 'location' in query_params:
            reviews = [review for review in reviews if review['Location'] == query_params['location'][0]]

        # Filter by start date
        if 'start_date' in query_params:
            start_date = datetime.strptime(query_params['start_date'][0], '%Y-%m-%d')
            reviews = [review for review in reviews if datetime.strptime(review['Timestamp'], TIMESTAMP_FORMAT) >= start_date]

        # Filter by end date
        if 'end_date' in query_params:
            end_date = datetime.strptime(query_params['end_date'][0], '%Y-%m-%d')
            reviews = [review for review in reviews if datetime.strptime(review['Timestamp'], TIMESTAMP_FORMAT) <= end_date]

        return reviews
    
    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        
        method = environ["REQUEST_METHOD"]

        if method == "GET":
            query_params = parse_qs(urlparse(environ['QUERY_STRING']).query)
            filtered_reviews = self.filter_reviews(reviews, query_params)

            # Sort reviews by compound sentiment score in descending order
            sorted_reviews = sorted(filtered_reviews, key=lambda x: x['sentiment']['compound'], reverse=True)
            
            response_body = json.dumps(sorted_reviews, indent=2).encode("utf-8")

            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            
            return [response_body]

        if method == "POST":
            try:
                request_body_size = int(environ.get("CONTENT_LENGTH", 0))
                request_body = environ["wsgi.input"].read(request_body_size)
                params = parse_qs(request_body.decode('utf-8'))

                if "Location" not in params or "ReviewBody" not in params:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "Location and ReviewBody are required fields."}).encode("utf-8")]

                # Additional validation for Location
                if params['Location'][0] not in ['San Diego, California', 'Denver, Colorado']:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "Invalid location."}).encode("utf-8")]

                review = {
                    'Location': params['Location'][0],
                    'ReviewBody': params['ReviewBody'][0],
                    'ReviewId': str(uuid.uuid4()),
                    'Timestamp': datetime.now().strftime(TIMESTAMP_FORMAT),
                    'sentiment': self.analyze_sentiment(params['ReviewBody'][0])
                }

                reviews.append(review)

                response_body = json.dumps(review, indent=2).encode("utf-8")

                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])

                return [response_body]
            
            except Exception as e:
                start_response("500 Internal Server Error", [("Content-Type", "application/json")])
                return [json.dumps({"error": str(e)}).encode("utf-8")]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
