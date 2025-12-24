"""
Tutorial 2: Evaluating a RAG System
Demonstrates how to evaluate a Retrieval-Augmented Generation system using RAGAS.
"""

# Standard library imports
import asyncio
from pathlib import Path

# Third-party imports
import pandas as pd
from dotenv import load_dotenv
from litellm import completion
from openai import AsyncOpenAI

# RAGAS imports
from ragas import experiment
from ragas.dataset import Dataset
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)


class SimpleRAGClient:
    """Simple RAG implementation with keyword-based retrieval."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        # Document corpus about Ragas
        self.documents = [
            "Ragas 0.3 is a library for evaluating LLM applications. It provides tools for measuring performance and quality.",
            "To install Ragas, you can install from pip using 'pip install ragas[examples]' or install from source.",
            "Ragas is organized around three main components: experiments for running evaluations, datasets for managing test data, and metrics for measuring performance.",
            "Ragas supports various metrics including faithfulness, answer relevancy, and context precision.",
            "The library integrates with popular LLM frameworks and provides a simple API for evaluation."
        ]

    def retrieve(self, query: str, top_k: int = 2):
        """Simple keyword-based retrieval."""
        # Simple scoring based on keyword overlap
        query_words = set(query.lower().split())
        scored_docs = []

        for doc in self.documents:
            doc_words = set(doc.lower().split())
            score = len(query_words.intersection(doc_words))
            scored_docs.append((score, doc))

        # Sort by score and return top_k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scored_docs[:top_k]]

    def query(self, query: str):
        """RAG pipeline: retrieve + generate."""
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query)
        context = "\n\n".join(retrieved_docs)

        # Generate answer using LLM
        prompt = f"""Answer the question based on the following context.

Context:
{context}

Question: {query}

Answer:"""

        messages = [{"role": "user", "content": prompt}]

        try:
            response = completion(
                model=self.model,
                messages=messages
            )
            answer = response.choices[0].message.content.strip()

            return {
                "answer": answer,
                "logs": f"Retrieved {len(retrieved_docs)} documents"
            }
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "logs": f"Error occurred: {str(e)}"
            }


# Initialize RAG client
rag_client = SimpleRAGClient()

# Create LLM for metrics
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Define evaluation metric
my_metric = DiscreteMetric(
    name="correctness",
    prompt="Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\nResponse: {response}\nGrading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)


@experiment()
async def run_experiment(row):
    """Run RAG experiment on a single row."""
    response = rag_client.query(row["query"])

    score = await my_metric.ascore(
        llm=llm,
        response=response.get("answer", " "),
        grading_notes=row["grading_notes"]
    )

    experiment_view = {
        **row,
        "response": response.get("answer", ""),
        "score": score.value,
        "log_file": response.get("logs", " "),
    }
    return experiment_view


if __name__ == "__main__":
    # Create test dataset
    samples = [
        {"query": "What is Ragas 0.3?", "grading_notes": "- Ragas 0.3 is a library for evaluating LLM applications."},
        {"query": "How to install Ragas?", "grading_notes": "- install from source  - install from pip using ragas[examples]"},
        {"query": "What are the main features of Ragas?", "grading_notes": "organised around - experiments - datasets - metrics."}
    ]
    dataset_file = DATASETS_DIR / "rag_test_dataset.csv"
    pd.DataFrame(samples).to_csv(dataset_file, index=False)

    # Load dataset
    df = pd.read_csv(dataset_file)
    dataset = Dataset.from_pandas(df, name="rag_queries", backend="local/csv", root_dir=str(DATASETS_DIR))

    # Run the experiment
    asyncio.run(run_experiment.arun(dataset, name="rag_evaluation"))
