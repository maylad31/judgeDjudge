from utils import call_openrouter
from string import Template
from pydantic import BaseModel
import csv
import argparse
from datetime import datetime
import os
import json
from dotenv import load_dotenv
from typing import Optional, Any, Dict, List, Tuple
from enum import Enum

load_dotenv()

PROMPT = Template("""You are a judge responsible for judging
which of the two responses best answers the task, given the rubrics.
\n
Task:
$task
\n
Rubrics: 
$rubrics
\n
Response1:
$response1
\n
Response2:
$response2
\n
Return which response was the best(Response1 or Response2).The provide a concise explanation.
""")

REPHRASE_PROMPT = Template("""You are provided with two texts. 
Rephrase them slightly but do not change their meaning. If their is code in any response, just add comments without modifying the code.
\n
Response1:
$response1
\n
Response2:
$response2
""")


class Response(str, Enum):
    RESPONSE1 = "response1"
    RESPONSE2 = "response2"


class Result(BaseModel):
    selected_response: str
    explanation: str


class Rephrase(BaseModel):
    response1_rephrased: str
    response2_rephrased: str


class Judge:
    def __init__(self, filepath: str | None = None) -> None:
        self.filepath = filepath
        self.consistency_score = 0
        self.position_score = 0
        self.verbosity_score = 0
        self.overall_score = 0

    def evaluate(
        self, model: str, openai_params: Optional[dict], provider_params: Optional[dict]
    ) -> Optional[dict]:
        """Evaluate model responses using CSV input and compute scoring metrics."""
        try:
            if openai_params is None:
                openai_params = {}
            if provider_params is None:
                provider_params = {}

            with open(self.filepath, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                required_fields = {
                    "task",
                    "rubrics",
                    "ideal_response",
                    "negative_response",
                }
                if not reader.fieldnames or not required_fields.issubset(
                    set(reader.fieldnames)
                ):
                    print("Invalid CSV: Missing required fields")
                    exit(1)

                results = []  # Collect all results for output file
                row_index = 0
                for row in reader:
                    if row_index > 0:
                        break
                    row_index += 1
                    task = row["task"]
                    rubrics = row["rubrics"]
                    ideal_response = row["ideal_response"]
                    negative_response = row["negative_response"]

                    row_consistency_score = 0
                    row_position_score = 0
                    row_verbosity_score = 0
                    row_explanations = []

                    for i in range(2):
                        score, explanation = self.check_consistency_score(
                            model,
                            task,
                            rubrics,
                            ideal_response,
                            negative_response,
                            openai_params,
                            provider_params,
                        )
                        row_consistency_score += score
                        row_explanations.append(
                            {
                                "type": "consistency",
                                "iteration": i + 1,
                                "score": score,
                                "explanation": explanation,
                            }
                        )

                    self.consistency_score += row_consistency_score

                    score, explanation = self.check_position_score(
                        model,
                        task,
                        rubrics,
                        ideal_response,
                        negative_response,
                        openai_params,
                        provider_params,
                    )
                    row_position_score += score
                    self.position_score += score
                    row_explanations.append(
                        {
                            "type": "position",
                            "score": score,
                            "explanation": explanation,
                        }
                    )

                    score, explanation = self.check_rephrase_score(
                        model,
                        task,
                        rubrics,
                        ideal_response,
                        negative_response,
                        openai_params,
                        provider_params,
                    )
                    row_verbosity_score += score
                    self.verbosity_score += score
                    row_explanations.append(
                        {
                            "type": "verbosity",
                            "score": score,
                            "explanation": explanation,
                        }
                    )

                    row_total = (
                        row_consistency_score + row_position_score + row_verbosity_score
                    )
                    self.overall_score += row_total

                    results.append(
                        {
                            "row_index": row_index,
                            "task": task,
                            "rubrics": rubrics,
                            "consistency_score": row_consistency_score,
                            "position_score": row_position_score,
                            "verbosity_score": row_verbosity_score,
                            "row_total": row_total,
                            "explanations": row_explanations,
                        }
                    )

                self.write_results_to_file(results, model)

                return {
                    "consistency_score": self.consistency_score,
                    "position_score": self.position_score,
                    "verbosity_score": self.verbosity_score,
                    "overall_score": self.overall_score,
                }

        except FileNotFoundError:
            print("File not found")
        except Exception as e:
            print(f"Error: {e}")

    def write_results_to_file(self, results: List[Dict[str, Any]], model: str) -> None:
        """Write results to a nicely formatted text file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # Text file with detailed explanations
        text_filename = (
            f"{output_dir}/judgment_results_{timestamp}_{model.replace('/', '_')}.txt"
        )
        with open(text_filename, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("JUDGMENT RESULTS\n")
            f.write(f"Model: {model}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            for result in results:
                f.write(f"Row #{result['row_index']}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Task:\n{result['task'][:200]}...\n\n")
                f.write(f"Rubrics:\n{result['rubrics'][:200]}...\n\n")

                f.write("Scores:\n")
                f.write(f"  Consistency Score: {result['consistency_score']}/2\n")
                f.write(f"  Position Score:    {result['position_score']}/1\n")
                f.write(f"  Verbosity Score:   {result['verbosity_score']}/1\n")
                f.write(f"  Row Total:         {result['row_total']}/4\n\n")

                f.write("Explanations:\n")
                for exp in result["explanations"]:
                    f.write(f"  [{exp['type'].upper()}] ")
                    if "iteration" in exp:
                        f.write(f"[Iteration {exp['iteration']}] ")
                    f.write(f"Score: {exp['score']}\n")
                    f.write(f"    {exp['explanation']}\n\n")

                f.write("\n")

            # Summary
            f.write("=" * 80 + "\n")
            f.write("SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Consistency Score: {self.consistency_score}\n")
            f.write(f"Position Score:    {self.position_score}\n")
            f.write(f"Verbosity Score:   {self.verbosity_score}\n")
            f.write(f"Overall Score:     {self.overall_score}\n")

        print(f"Results written to {text_filename}")

    def check_consistency_score(
        self,
        model: str,
        task: str,
        rubrics: str,
        ideal_response: str,
        negative_response: str,
        openai_params: Dict[str, Any],
        provider_params: Dict[str, Any],
    ) -> Tuple[int, str]:
        """Check if the model correctly identifies the ideal response (Response1)."""
        score = 0
        prompt = PROMPT.substitute(
            task=task,
            rubrics=rubrics,
            response1=ideal_response,
            response2=negative_response,
        )
        result = call_openrouter(
            prompt=prompt,
            model=model,
            response_model=Result,
            openai_params=openai_params,
            provider_params=provider_params,
        )
        print(result)
        if result.selected_response.lower() == Response.RESPONSE1:
            score = 1
        return score, result.explanation

    def check_position_score(
        self,
        model: str,
        task: str,
        rubrics: str,
        ideal_response: str,
        negative_response: str,
        openai_params: Dict[str, Any],
        provider_params: Dict[str, Any],
    ) -> Tuple[int, str]:
        """Check if the model correctly identifies the ideal response when it's Response2."""
        score = 0
        prompt = PROMPT.substitute(
            task=task,
            rubrics=rubrics,
            response1=negative_response,
            response2=ideal_response,
        )
        result = call_openrouter(
            prompt=prompt,
            model=model,
            response_model=Result,
            openai_params=openai_params,
            provider_params=provider_params,
        )
        if result.selected_response.lower() == Response.RESPONSE2:
            score = 1
        return score, result.explanation

    def check_rephrase_score(
        self,
        model: str,
        task: str,
        rubrics: str,
        ideal_response: str,
        negative_response: str,
        openai_params: Dict[str, Any],
        provider_params: Dict[str, Any],
    ) -> Tuple[int, str]:
        """Check consistency when both rephrased responses should be equivalent."""
        score = 0
        rephrase_prompt = REPHRASE_PROMPT.substitute(
            response1=ideal_response, response2=negative_response
        )
        rephrased = call_openrouter(
            prompt=rephrase_prompt,
            model=model,
            response_model=Rephrase,
            openai_params=openai_params,
            provider_params=provider_params,
        )
        prompt = PROMPT.substitute(
            task=task,
            rubrics=rubrics,
            response1=rephrased.response1_rephrased,
            response2=rephrased.response2_rephrased,
        )
        result = call_openrouter(
            prompt=prompt,
            model=model,
            response_model=Result,
            openai_params=openai_params,
            provider_params=provider_params,
        )
        if result.selected_response.lower() == Response.RESPONSE1:
            score = 1
        return score, result.explanation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Judge model responses against rubrics and generate detailed scores"
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model identifier (e.g., openai/gpt-4o, openrouter/mistralai)",
    )
    parser.add_argument(
        "--input",
        "-i",
        default="input.csv",
        help="Path to input CSV file (default: input.csv)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="output",
        help="Directory for output files (default: output)",
    )
    parser.add_argument(
        "--openai-params",
        type=str,
        default=None,
        help='JSON string for OpenAI-style parameters (e.g., \'{"max_tokens": 1024, "temperature": 0.7}\')',
    )

    parser.add_argument(
        "--provider-params",
        type=str,
        default=None,
        help="JSON string for provider-specific parameters (e.g., '{\"stream\": false}')",
    )

    args = parser.parse_args()

    # Build openai_params only if specified
    openai_params = {}
    if args.openai_params is not None:
        try:
            openai_params = json.loads(args.openai_params)
        except json.JSONDecodeError as e:
            print(f"Error parsing openai params: {e}")
            exit(1)
    # Parse provider params if provided
    provider_params = {}
    if args.provider_params is not None:
        try:
            provider_params = json.loads(args.provider_params)
        except json.JSONDecodeError as e:
            print(f"Error parsing provider params: {e}")
            exit(1)

    judge = Judge(filepath=args.input)
    results = judge.evaluate(args.model, openai_params, provider_params)

    if results:
        print("\n=== Final Scores ===")
        print(f"Consistency Score: {results['consistency_score']}")
        print(f"Position Score:    {results['position_score']}")
        print(f"Verbosity Score:   {results['verbosity_score']}")
        print(f"Overall Score:     {results['overall_score']}")
