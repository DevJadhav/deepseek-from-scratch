"""
ANE Task Accuracy Evaluation Module

Provides evaluation on standard benchmarks:
- MMLU (Massive Multitask Language Understanding)
- HumanEval (code generation)
- Custom task evaluation
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TaskConfig:
    """Configuration for task evaluation."""

    # Maximum sequence length
    max_seq_len: int = 2048

    # Number of few-shot examples
    num_shots: int = 5

    # Batch size for evaluation
    batch_size: int = 1

    # Device for computation
    device: str = "cpu"

    # Maximum samples to evaluate (None = all)
    max_samples: int | None = None

    # Verbose output
    verbose: bool = False


@dataclass
class TaskResult:
    """Result of task evaluation."""

    # Task name
    task_name: str

    # Accuracy (0-1)
    accuracy: float

    # Number correct
    num_correct: int

    # Total samples
    num_total: int

    # Per-category results (for MMLU)
    category_results: dict[str, dict[str, float]] = field(default_factory=dict)

    # Additional metrics
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "task_name": self.task_name,
            "accuracy": self.accuracy,
            "num_correct": self.num_correct,
            "num_total": self.num_total,
            "category_results": self.category_results,
            "extra_metrics": self.extra_metrics,
        }


# MMLU Subjects
MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

MMLU_CATEGORIES = {
    "STEM": [
        "abstract_algebra",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "Social Sciences": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
    "Other": [
        "anatomy",
        "business_ethics",
        "clinical_knowledge",
        "college_medicine",
        "global_facts",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
    ],
}


def format_mmlu_prompt(
    question: str,
    choices: list[str],
    subject: str,
    few_shot_examples: list[dict] | None = None,
) -> str:
    """
    Format MMLU question as prompt.

    Args:
        question: The question text
        choices: List of answer choices
        subject: Subject name
        few_shot_examples: Optional few-shot examples

    Returns:
        Formatted prompt string
    """
    prompt = f"The following are multiple choice questions about {subject.replace('_', ' ')}.\n\n"

    # Add few-shot examples
    if few_shot_examples:
        for example in few_shot_examples:
            prompt += f"Question: {example['question']}\n"
            for i, choice in enumerate(example["choices"]):
                prompt += f"{chr(65 + i)}. {choice}\n"
            prompt += f"Answer: {example['answer']}\n\n"

    # Add target question
    prompt += f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65 + i)}. {choice}\n"
    prompt += "Answer:"

    return prompt


def load_mmlu_data(
    subject: str,
    split: str = "test",
    data_dir: str | None = None,
) -> list[dict]:
    """
    Load MMLU data for a subject.

    Args:
        subject: Subject name
        split: Dataset split
        data_dir: Optional data directory

    Returns:
        List of question dictionaries
    """
    # Try huggingface datasets
    try:
        from datasets import load_dataset

        dataset = load_dataset("cais/mmlu", subject, split=split)
        questions = []
        for item in dataset:
            questions.append(
                {
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": chr(65 + item["answer"]),  # Convert to A/B/C/D
                }
            )
        return questions
    except (ImportError, Exception):
        pass

    # Return sample data for testing
    return [
        {
            "question": "What is 2 + 2?",
            "choices": ["3", "4", "5", "6"],
            "answer": "B",
        },
        {
            "question": "What is the capital of France?",
            "choices": ["London", "Paris", "Berlin", "Madrid"],
            "answer": "B",
        },
    ]


def evaluate_mmlu_question(
    model: nn.Module,
    tokenizer,
    question: dict,
    subject: str,
    few_shot_examples: list[dict] | None = None,
    device: str = "cpu",
) -> tuple[str, bool]:
    """
    Evaluate a single MMLU question.

    Args:
        model: Language model
        tokenizer: Tokenizer
        question: Question dictionary
        subject: Subject name
        few_shot_examples: Few-shot examples
        device: Device for computation

    Returns:
        Tuple of (predicted answer, is_correct)
    """
    prompt = format_mmlu_prompt(
        question["question"],
        question["choices"],
        subject,
        few_shot_examples,
    )

    # Tokenize
    if hasattr(tokenizer, "encode"):
        input_ids = tokenizer.encode(prompt)
    else:
        input_ids = tokenizer(prompt)

    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids, dtype=torch.long)

    input_ids = input_ids.unsqueeze(0).to(device)

    # Get model output
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)

    # Extract logits
    if isinstance(outputs, torch.Tensor):
        logits = outputs
    elif hasattr(outputs, "logits"):
        logits = outputs.logits
    elif isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        return "A", False

    # Get next token probabilities
    next_token_logits = logits[0, -1, :]

    # Get probabilities for A, B, C, D tokens
    choice_tokens = []
    for letter in ["A", "B", "C", "D"]:
        if hasattr(tokenizer, "encode"):
            token_id = tokenizer.encode(letter)
            if isinstance(token_id, list):
                token_id = token_id[-1] if token_id else 0
        else:
            token_id = ord(letter)
        choice_tokens.append(token_id)

    choice_logits = next_token_logits[choice_tokens]
    predicted_idx = choice_logits.argmax().item()
    predicted = chr(65 + predicted_idx)

    is_correct = predicted == question["answer"]

    return predicted, is_correct


class MMLUEvaluator:
    """
    MMLU (Massive Multitask Language Understanding) evaluator.

    Example:
        evaluator = MMLUEvaluator(config)

        # Evaluate on all subjects
        result = evaluator.evaluate(model, tokenizer)
        print(f"MMLU Accuracy: {result.accuracy:.2%}")

        # Evaluate on specific subjects
        result = evaluator.evaluate(model, tokenizer, subjects=["abstract_algebra"])
    """

    def __init__(self, config: TaskConfig | None = None):
        """Initialize evaluator."""
        self.config = config or TaskConfig()

    def evaluate(
        self,
        model: nn.Module,
        tokenizer,
        subjects: list[str] | None = None,
    ) -> TaskResult:
        """
        Evaluate model on MMLU.

        Args:
            model: Language model
            tokenizer: Tokenizer
            subjects: Specific subjects to evaluate (None = all)

        Returns:
            TaskResult with accuracy metrics
        """
        subjects = subjects or MMLU_SUBJECTS
        device = torch.device(self.config.device)
        model = model.to(device)

        total_correct = 0
        total_questions = 0
        category_results: dict[str, dict[str, float]] = {}
        subject_results: dict[str, dict[str, float]] = {}

        for subject in subjects:
            questions = load_mmlu_data(subject)

            if self.config.max_samples:
                questions = questions[: self.config.max_samples]

            # Get few-shot examples (from dev set)
            few_shot = load_mmlu_data(subject, split="dev")[: self.config.num_shots]

            correct = 0
            for q in questions:
                _, is_correct = evaluate_mmlu_question(
                    model, tokenizer, q, subject, few_shot, self.config.device
                )
                if is_correct:
                    correct += 1

            accuracy = correct / len(questions) if questions else 0.0
            subject_results[subject] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": len(questions),
            }

            total_correct += correct
            total_questions += len(questions)

            if self.config.verbose:
                print(f"{subject}: {accuracy:.2%} ({correct}/{len(questions)})")

        # Aggregate by category
        for category, category_subjects in MMLU_CATEGORIES.items():
            cat_correct = sum(
                subject_results.get(s, {}).get("correct", 0)
                for s in category_subjects
                if s in subject_results
            )
            cat_total = sum(
                subject_results.get(s, {}).get("total", 0)
                for s in category_subjects
                if s in subject_results
            )
            if cat_total > 0:
                category_results[category] = {
                    "accuracy": cat_correct / cat_total,
                    "correct": cat_correct,
                    "total": cat_total,
                }

        return TaskResult(
            task_name="MMLU",
            accuracy=total_correct / total_questions if total_questions > 0 else 0.0,
            num_correct=total_correct,
            num_total=total_questions,
            category_results=category_results,
            extra_metrics={"subject_results": subject_results},
        )


def load_humaneval_data(data_dir: str | None = None) -> list[dict]:
    """
    Load HumanEval dataset.

    Args:
        data_dir: Optional data directory

    Returns:
        List of problem dictionaries
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset("openai/openai_humaneval", split="test")
        problems = []
        for item in dataset:
            problems.append(
                {
                    "task_id": item["task_id"],
                    "prompt": item["prompt"],
                    "canonical_solution": item["canonical_solution"],
                    "test": item["test"],
                    "entry_point": item["entry_point"],
                }
            )
        return problems
    except (ImportError, Exception):
        pass

    # Return sample data for testing
    return [
        {
            "task_id": "HumanEval/0",
            "prompt": 'def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    """\n',
            "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False\n",
            "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n",
            "entry_point": "has_close_elements",
        },
    ]


def execute_code_safely(
    code: str,
    test_code: str,
    timeout: float = 5.0,
) -> tuple[bool, str]:
    """
    Execute code safely with timeout.

    Args:
        code: Code to execute
        test_code: Test code to run
        timeout: Timeout in seconds

    Returns:
        Tuple of (passed, error_message)
    """
    import multiprocessing
    import traceback

    def run_tests(code: str, test_code: str, result_queue):
        try:
            # Create execution namespace
            namespace: dict[str, Any] = {}

            # Add common imports
            exec("from typing import List, Optional, Tuple, Dict, Any", namespace)

            # Execute the code
            exec(code, namespace)

            # Execute tests
            exec(test_code, namespace)

            result_queue.put((True, ""))
        except Exception as e:
            result_queue.put((False, traceback.format_exc()))

    result_queue: multiprocessing.Queue[tuple[bool, str]] = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=run_tests, args=(code, test_code, result_queue)
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False, "Timeout"

    if result_queue.empty():
        return False, "No result"

    return result_queue.get()


class HumanEvalEvaluator:
    """
    HumanEval evaluator for code generation.

    Example:
        evaluator = HumanEvalEvaluator(config)

        # Evaluate on HumanEval
        result = evaluator.evaluate(model, tokenizer)
        print(f"HumanEval pass@1: {result.accuracy:.2%}")
    """

    def __init__(self, config: TaskConfig | None = None):
        """Initialize evaluator."""
        self.config = config or TaskConfig()

    def generate_completion(
        self,
        model: nn.Module,
        tokenizer,
        prompt: str,
        max_tokens: int = 256,
    ) -> str:
        """
        Generate code completion.

        Args:
            model: Language model
            tokenizer: Tokenizer
            prompt: Code prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated completion
        """
        device = torch.device(self.config.device)

        # Tokenize
        if hasattr(tokenizer, "encode"):
            input_ids = tokenizer.encode(prompt)
        else:
            input_ids = tokenizer(prompt)

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        input_ids = input_ids.unsqueeze(0).to(device)

        model.eval()
        generated_ids = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = model(generated_ids)

                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    break

                next_token_logits = logits[0, -1, :]
                next_token = next_token_logits.argmax().unsqueeze(0).unsqueeze(0)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Check for end of generation
                # This is simplified - real implementation would check for EOS token

        # Decode
        generated_tokens = generated_ids[0, input_ids.shape[1] :]
        if hasattr(tokenizer, "decode"):
            completion = tokenizer.decode(generated_tokens.tolist())
        else:
            completion = "".join(chr(t) for t in generated_tokens.tolist())

        # Extract just the function body
        # Stop at newlines that indicate end of function
        lines = completion.split("\n")
        result_lines = []
        for line in lines:
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                if result_lines:  # Function body has ended
                    break
            result_lines.append(line)

        return "\n".join(result_lines)

    def evaluate(
        self,
        model: nn.Module,
        tokenizer,
        num_samples: int | None = None,
    ) -> TaskResult:
        """
        Evaluate model on HumanEval.

        Args:
            model: Language model
            tokenizer: Tokenizer
            num_samples: Number of samples to evaluate

        Returns:
            TaskResult with pass@1 accuracy
        """
        device = torch.device(self.config.device)
        model = model.to(device)

        problems = load_humaneval_data()

        if num_samples:
            problems = problems[:num_samples]
        elif self.config.max_samples:
            problems = problems[: self.config.max_samples]

        passed = 0
        total = len(problems)
        problem_results = {}

        for problem in problems:
            task_id = problem["task_id"]
            prompt = problem["prompt"]
            test = problem["test"]
            entry_point = problem["entry_point"]

            # Generate completion
            completion = self.generate_completion(model, tokenizer, prompt)

            # Combine prompt and completion
            full_code = prompt + completion

            # Create test code
            test_code = test + f"\ncheck({entry_point})"

            # Execute and check
            success, error = execute_code_safely(full_code, test_code)

            if success:
                passed += 1

            problem_results[task_id] = {
                "passed": success,
                "error": error if not success else "",
            }

            if self.config.verbose:
                status = "✓" if success else "✗"
                print(f"{task_id}: {status}")

        return TaskResult(
            task_name="HumanEval",
            accuracy=passed / total if total > 0 else 0.0,
            num_correct=passed,
            num_total=total,
            extra_metrics={"problem_results": problem_results},
        )


class TaskEvaluator:
    """
    Unified task evaluator for multiple benchmarks.

    Example:
        evaluator = TaskEvaluator(config)

        # Evaluate on MMLU
        mmlu_result = evaluator.evaluate_mmlu(model, tokenizer)

        # Evaluate on HumanEval
        humaneval_result = evaluator.evaluate_humaneval(model, tokenizer)

        # Run all evaluations
        results = evaluator.evaluate_all(model, tokenizer)
    """

    def __init__(self, config: TaskConfig | None = None):
        """Initialize evaluator."""
        self.config = config or TaskConfig()
        self.mmlu = MMLUEvaluator(config)
        self.humaneval = HumanEvalEvaluator(config)
        self.results: list[TaskResult] = []

    def evaluate_mmlu(
        self,
        model: nn.Module,
        tokenizer,
        subjects: list[str] | None = None,
    ) -> TaskResult:
        """Evaluate on MMLU."""
        result = self.mmlu.evaluate(model, tokenizer, subjects)
        self.results.append(result)
        return result

    def evaluate_humaneval(
        self,
        model: nn.Module,
        tokenizer,
    ) -> TaskResult:
        """Evaluate on HumanEval."""
        result = self.humaneval.evaluate(model, tokenizer)
        self.results.append(result)
        return result

    def evaluate_all(
        self,
        model: nn.Module,
        tokenizer,
    ) -> dict[str, TaskResult]:
        """
        Run all evaluations.

        Args:
            model: Language model
            tokenizer: Tokenizer

        Returns:
            Dictionary of task name to result
        """
        results = {}

        if self.config.verbose:
            print("Evaluating MMLU...")
        results["mmlu"] = self.evaluate_mmlu(model, tokenizer)

        if self.config.verbose:
            print("\nEvaluating HumanEval...")
        results["humaneval"] = self.evaluate_humaneval(model, tokenizer)

        return results

    def get_summary(self) -> dict:
        """Get summary of all results."""
        return {
            "num_tasks": len(self.results),
            "results": {r.task_name: r.to_dict() for r in self.results},
        }

    def save_results(self, filename: str):
        """Save results to JSON file."""
        with open(filename, "w") as f:
            json.dump(self.get_summary(), f, indent=2)
