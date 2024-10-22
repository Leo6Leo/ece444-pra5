import time
from datetime import datetime
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns


def test_prediction(text, num_calls=100):
    """
    Test the prediction API with a given text multiple times and record latencies

    Args:
        text (str): Text to test
        num_calls (int): Number of API calls to make

    Returns:
        list: List of latencies in milliseconds
    """
    url = (
        "http://flask-canada-env.eba-pdr92yfm.ca-central-1.elasticbeanstalk.com/predict"
    )
    latencies = []
    predictions = []
    timestamps = []

    for _ in range(num_calls):
        start_time = time.time()
        response = requests.post(url, json={"text": text})
        end_time = time.time()

        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        prediction = response.json()["prediction"]

        latencies.append(latency)
        predictions.append(prediction)
        timestamps.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))

    return latencies, predictions, timestamps


# Test cases
test_cases = {
    "fake_news_1": "BREAKING: Aliens have officially made contact with Earth government officials in secret meeting",
    "fake_news_2": "Scientists discover that drinking coffee grants immortality, governments suppress information",
    "real_news_1": "The president made an official statement today.",
    "real_news_2": "The professor made an official statement today.",
}

# Store results
all_results = []

# Run tests for each case
for case_name, text in test_cases.items():
    print(f"Testing {case_name}...")
    latencies, predictions, timestamps = test_prediction(text)

    # Create DataFrame for this test case
    df = pd.DataFrame(
        {
            "test_case": case_name,
            "timestamp": timestamps,
            "latency_ms": latencies,
            "prediction": predictions,
        }
    )

    all_results.append(df)

    # Print functional test results
    print(f"Most common prediction: {max(set(predictions), key=predictions.count)}")
    print(f"Average latency: {mean(latencies):.2f}ms\n")

# Combine all results
results_df = pd.concat(all_results, ignore_index=True)

# Save results to CSV
results_df.to_csv("api_test_results.csv", index=False)

# Create boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x="test_case", y="latency_ms", data=results_df)
plt.xticks(rotation=45)
plt.title("API Latency Distribution by Test Case")
plt.xlabel("Test Case")
plt.ylabel("Latency (ms)")
plt.tight_layout()
plt.savefig("latency_boxplot.png")
plt.close()

# Print summary statistics
print("\nSummary Statistics:")
summary = (
    results_df.groupby("test_case")["latency_ms"]
    .agg(["mean", "min", "max", "std"])
    .round(2)
)
print(summary)
