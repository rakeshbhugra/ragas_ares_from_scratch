"""
Tutorial 1: Evaluating a Prompt
Demonstrates how to evaluate a simple sentiment classification prompt using RAGAS.
"""

# Standard library imports
import asyncio
from pathlib import Path

# Third-party imports
import pandas as pd
from dotenv import load_dotenv
from litellm import completion

# RAGAS imports
from ragas import experiment
from ragas.dataset import Dataset
from ragas.metrics import discrete_metric
from ragas.metrics.result import MetricResult

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)


def run_prompt(text: str, model: str) -> str:
    """
    Classify sentiment of a movie review using an LLM.

    This function creates a prompt that asks the LLM to classify a review's sentiment,
    sends it to the model, and extracts the classification result.

    Args:
        text: The review text to classify (e.g., "I loved this movie!")
        model: The LLM model identifier (e.g., "gpt-4o-mini")

    Returns:
        str: Either 'positive' or 'negative' based on the review sentiment

    Example:
        >>> run_prompt("This movie was amazing!", "gpt-4o-mini")
        'positive'
    """
    # Create a clear, constrained prompt asking for one-word classification
    prompt = f"""Classify the sentiment of the following review as 'positive' or 'negative'.
Only respond with one word: either 'positive' or 'negative'.

Review: {text}

Sentiment:"""

    # Format the message for the LLM API
    messages = [{"role": "user", "content": prompt}]

    # Call the LLM via litellm (supports multiple providers)
    response = completion(
        model=model,
        messages=messages
    )

    # Extract the response text and normalize to lowercase
    result = response.choices[0].message.content.strip().lower()

    # Handle cases where the model might add extra text like "sentiment: positive"
    # We just check if the sentiment word is in the response
    if "positive" in result:
        return "positive"
    elif "negative" in result:
        return "negative"

    # Return as-is if neither keyword found (edge case)
    return result


@discrete_metric(name="accuracy", allowed_values=["pass", "fail"])
def my_metric(prediction: str, actual: str):
    """
    Evaluate if the prediction matches the actual label.

    This is a simple exact-match metric that returns "pass" if the prediction
    equals the ground truth, and "fail" otherwise. The @discrete_metric decorator
    converts this into a RAGAS-compatible metric.

    Args:
        prediction: The model's predicted sentiment ("positive" or "negative")
        actual: The ground truth label ("positive" or "negative")

    Returns:
        MetricResult: Contains "pass" if match, "fail" if mismatch
    """
    # Simple exact string comparison - return pass/fail based on match
    return MetricResult(value="pass", reason="") if prediction == actual else MetricResult(value="fail", reason="")


@experiment()
async def run_experiment(row, model):
    """
    Run the evaluation experiment on a single test example.

    This function is called once for each row in the dataset. It:
    1. Runs the prompt on the review text
    2. Scores the response against the ground truth
    3. Returns all data including the result for tracking

    The @experiment() decorator integrates this with RAGAS's experiment tracking.

    Args:
        row: Dictionary containing "text" (review) and "label" (ground truth)
        model: The LLM model to use for classification

    Returns:
        dict: Complete experiment view with input, output, and score
    """
    # Step 1: Get the model's prediction for this review
    response = run_prompt(row["text"], model=model)

    # Step 2: Score the prediction against ground truth
    score = my_metric.score(
        prediction=response,
        actual=row["label"]
    )

    # Step 3: Combine all data for experiment tracking
    experiment_view = {
        **row,  # Original data (text, label)
        "response": response,  # Model's prediction
        "score": score.value,  # Pass/fail result
    }
    return experiment_view


if __name__ == "__main__":
    # ============================================================================
    # STEP 1: Create test dataset with movie reviews
    # ============================================================================
    # Define a small set of movie reviews with ground truth sentiment labels
    samples = [
        {"text": "I loved the movie! It was fantastic.", "label": "positive"},
        {"text": "The movie was terrible and boring.", "label": "negative"},
        {"text": "It was an average film, nothing special.", "label": "positive"},
        {"text": "Absolutely amazing! Best movie of the year.", "label": "positive"}
    ]

    # Save to CSV file in the project's datasets directory
    dataset_file = DATASETS_DIR / "test_dataset.csv"
    pd.DataFrame(samples).to_csv(dataset_file, index=False)

    # ============================================================================
    # STEP 2: Load dataset into RAGAS format
    # ============================================================================
    # Read the CSV back and convert to RAGAS Dataset object
    # This enables RAGAS's experiment tracking and backend storage
    df = pd.read_csv(dataset_file)
    dataset = Dataset.from_pandas(
        df,
        name="movie_reviews",  # Name for tracking
        backend="local/csv",   # Store results as CSV files
        root_dir=str(DATASETS_DIR)  # Where to save experiment results
    )

    # ============================================================================
    # STEP 3: Run the evaluation experiment
    # ============================================================================
    # This will:
    # - Run run_experiment() on each row asynchronously
    # - Collect all results
    # - Save to datasets/experiments/prompt_evaluation.csv (with consistent name)
    results = asyncio.run(run_experiment.arun(dataset, name="prompt_evaluation", model="gpt-4o-mini"))

    # ============================================================================
    # STEP 4: Display results
    # ============================================================================
    # Convert results to pandas DataFrame for easy viewing
    results_df = results.to_pandas()

    # Print full results table
    print("\n=== Experiment Results ===")
    print(results_df.to_string())

    # Calculate and display overall accuracy
    print("\n=== Summary ===")
    pass_count = (results_df['score'] == 'pass').sum()
    total = len(results_df)
    print(f"Accuracy: {pass_count}/{total} = {pass_count/total*100:.1f}%")

    # Show individual predictions with pass/fail indicators
    print("\n=== Individual Results ===")
    for idx, row in results_df.iterrows():
        status = "✓" if row['score'] == 'pass' else "✗"
        print(f"{status} Text: {row['text'][:50]}...")
        print(f"  Expected: {row['label']}, Got: {row['response']}")
