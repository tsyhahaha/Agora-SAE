"""Paper-aligned evaluation utilities for the MATH500 reproduction path."""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from agora_sae.data.mixed_source import MixedTokenSource
from agora_sae.data.reasoning_steps import (
    ActivationPointSelector,
    DelimiterStepSegmenter,
    ReasoningStep,
)
from agora_sae.judge_transport import post_json_with_retry
from agora_sae.jsonl_resume import load_jsonl_records, prepare_jsonl_output


SOLUTION_HEADER_PATTERN = re.compile(
    r"(?:Solution|Answer|解答|答案)\s*[:：]\s*",
    re.IGNORECASE,
)
FINAL_ANSWER_PATTERN = re.compile(
    r"(?:Final Answer|Answer)\s*[:：]\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class ReasoningSample:
    """Structured reasoning sample used by the paper-style evaluation pipeline."""

    sample_id: str
    question: str
    response: str
    full_text: str
    response_start_char: int
    steps: List[ReasoningStep]
    reference_answer: Optional[str] = None


class StepJudge:
    """Interface for step-level behavior labeling."""

    def classify_step(
        self,
        question: str,
        response: str,
        step_text: str,
        previous_step: Optional[str] = None,
        next_step: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        raise NotImplementedError


class HeuristicStepJudge(StepJudge):
    """Lightweight fallback labeler when an external judge is unavailable."""

    REFLECTION_PATTERNS = tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bwait\b",
            r"double[- ]check",
            r"check again",
            r"just to make sure",
            r"let me verify",
            r"verify that",
            r"let me think again",
            r"reconsider",
            r"does that make sense",
            r"i should make sure",
        )
    )
    BACKTRACKING_PATTERNS = tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\balternatively\b",
            r"another approach",
            r"instead",
            r"let'?s try",
            r"that (?:won't|will not) work",
            r"different approach",
            r"back up",
            r"start over",
            r"switch to",
            r"rewrite the plan",
        )
    )

    def __init__(self):
        self.reflection_patterns = tuple(self.REFLECTION_PATTERNS)
        self.backtracking_patterns = tuple(self.BACKTRACKING_PATTERNS)

    def classify_step(
        self,
        question: str,
        response: str,
        step_text: str,
        previous_step: Optional[str] = None,
        next_step: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        reflection_hits = sum(bool(pattern.search(step_text)) for pattern in self.reflection_patterns)
        backtracking_hits = sum(bool(pattern.search(step_text)) for pattern in self.backtracking_patterns)

        if reflection_hits > backtracking_hits and reflection_hits > 0:
            return "reflection", "heuristic cue match"
        if backtracking_hits > reflection_hits and backtracking_hits > 0:
            return "backtracking", "heuristic cue match"
        return "other", "no heuristic cue match"


class OpenAIJudge(StepJudge):
    """LLM-as-a-judge classifier using the OpenAI Responses API."""

    API_URL = "https://api.openai.com/v1/responses"

    def __init__(
        self,
        model: str = "gpt-5",
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 5,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI judge.")

    def classify_step(
        self,
        question: str,
        response: str,
        step_text: str,
        previous_step: Optional[str] = None,
        next_step: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        system_prompt = (
            "You are labeling a reasoning step from a math solution. "
            "Use exactly one label: reflection, backtracking, or other. "
            "Reflection means re-checking or re-examining earlier reasoning. "
            "Backtracking means abandoning the current line of reasoning and switching to a new one. "
            "Other means the step is neither reflection nor backtracking."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Full response:\n{response}\n\n"
            f"Previous step:\n{previous_step or '<none>'}\n\n"
            f"Current step:\n{step_text}\n\n"
            f"Next step:\n{next_step or '<none>'}\n\n"
            "Return JSON with keys label and rationale."
        )
        schema = {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": ["reflection", "backtracking", "other"],
                },
                "rationale": {"type": "string"},
            },
            "required": ["label", "rationale"],
            "additionalProperties": False,
        }
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "behavior_label",
                    "schema": schema,
                    "strict": True,
                }
            },
            "max_output_tokens": 200,
        }

        response_json = self._request(payload)
        parsed = self._parse_response_json(response_json)
        return parsed["label"], parsed.get("rationale")

    def _request(self, payload: Dict) -> Dict:
        return post_json_with_retry(
            url=self.API_URL,
            payload=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
            max_retries=self.max_retries,
            provider_name="OpenAI judge",
        )

    @staticmethod
    def _parse_response_json(response_json: Dict) -> Dict[str, str]:
        raw_text = response_json.get("output_text")
        if not raw_text:
            texts = []
            for item in response_json.get("output", []):
                for content in item.get("content", []):
                    text_value = content.get("text")
                    if isinstance(text_value, dict):
                        text_value = text_value.get("value")
                    if content.get("type") in {"output_text", "text"} and text_value:
                        texts.append(text_value)
            raw_text = "\n".join(texts).strip()

        if not raw_text:
            raise ValueError("OpenAI response did not contain output text.")

        parsed = json.loads(raw_text)
        label = parsed.get("label")
        if label not in {"reflection", "backtracking", "other"}:
            raise ValueError(f"Unexpected label from judge: {label}")
        return parsed


class MinimaxJudge(StepJudge):
    """LLM-as-a-judge classifier using the MiniMax OpenAI-compatible API."""

    DEFAULT_BASE_URL = "https://api.minimax.io/v1"
    DEFAULT_MAX_TOKENS = 128

    def __init__(
        self,
        model: str = "MiniMax-M2.5",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 5,
        max_output_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning_split: bool = True,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        self.base_url = (base_url or os.environ.get("MINIMAX_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        self.reasoning_split = reasoning_split
        if not self.api_key:
            raise ValueError("MINIMAX_API_KEY is required for the MiniMax judge.")

    def classify_step(
        self,
        question: str,
        response: str,
        step_text: str,
        previous_step: Optional[str] = None,
        next_step: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        system_prompt = (
            "You are labeling a reasoning step from a math solution. "
            "Use exactly one label: reflection, backtracking, or other. "
            "Reflection means re-checking or re-examining earlier reasoning. "
            "Backtracking means abandoning the current line of reasoning and switching to a new one. "
            "Other means the step is neither reflection nor backtracking. "
            "Return only valid JSON with keys label and rationale."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Full response:\n{response}\n\n"
            f"Previous step:\n{previous_step or '<none>'}\n\n"
            f"Current step:\n{step_text}\n\n"
            f"Next step:\n{next_step or '<none>'}\n\n"
            "Return JSON with keys label and rationale."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_output_tokens,
        }
        if self.reasoning_split:
            # MiniMax documents this as the raw request-body field used by the
            # OpenAI-compatible API to move thinking content out of message.content.
            payload["reasoning_split"] = True

        response_json = self._request(payload)
        parsed = self._parse_response_json(response_json)
        return parsed["label"], parsed.get("rationale")

    def _request(self, payload: Dict) -> Dict:
        return post_json_with_retry(
            url=f"{self.base_url}/chat/completions",
            payload=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
            max_retries=self.max_retries,
            provider_name="MiniMax judge",
        )

    @classmethod
    def _parse_response_json(cls, response_json: Dict) -> Dict[str, str]:
        choices = response_json.get("choices") or []
        if not choices:
            raise ValueError("MiniMax response did not contain any choices.")

        message = choices[0].get("message") or {}
        raw_text = cls._extract_message_text(message.get("content"))
        if not raw_text:
            raise ValueError("MiniMax response did not contain message content.")

        return cls._parse_label_payload(raw_text)

    @staticmethod
    def _extract_message_text(content) -> Optional[str]:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("value") or item.get("content")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            return "\n".join(parts).strip() or None
        return None

    @staticmethod
    def _parse_label_payload(raw_text: str) -> Dict[str, str]:
        candidates = [raw_text.strip()]

        fenced_match = JSON_BLOCK_PATTERN.search(raw_text)
        if fenced_match:
            candidates.append(fenced_match.group(1).strip())

        start_index = raw_text.find("{")
        end_index = raw_text.rfind("}")
        if start_index != -1 and end_index != -1 and end_index > start_index:
            candidates.append(raw_text[start_index : end_index + 1].strip())

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            label = parsed.get("label")
            if label not in {"reflection", "backtracking", "other"}:
                continue
            rationale = parsed.get("rationale")
            if rationale is not None:
                rationale = str(rationale)
            return {"label": label, "rationale": rationale}

        raise ValueError(f"Could not parse MiniMax judge response as label JSON: {raw_text}")


def get_step_judge(
    judge: str,
    judge_model: Optional[str] = None,
    *,
    timeout: int = 60,
    max_retries: int = 5,
    minimax_max_output_tokens: int = MinimaxJudge.DEFAULT_MAX_TOKENS,
    minimax_reasoning_split: bool = True,
) -> StepJudge:
    """Build the requested step judge."""
    if judge == "heuristic":
        return HeuristicStepJudge()
    if judge == "openai":
        return OpenAIJudge(
            model=judge_model or "gpt-5",
            timeout=timeout,
            max_retries=max_retries,
        )
    if judge == "minimax":
        return MinimaxJudge(
            model=judge_model or "MiniMax-M2.5",
            timeout=timeout,
            max_retries=max_retries,
            max_output_tokens=minimax_max_output_tokens,
            reasoning_split=minimax_reasoning_split,
        )
    raise ValueError(f"Unsupported judge: {judge}")


def build_prompt(question: str, prompt_template: str) -> str:
    """Render a generation prompt from a question string."""
    return prompt_template.format(question=question.strip())


def normalize_answer(text: Optional[str]) -> Optional[str]:
    """Normalize an extracted answer for rough string-based comparison."""
    if text is None:
        return None
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return normalized or None


def extract_final_answer(text: Optional[str]) -> Optional[str]:
    """Extract a compact final-answer string from a response."""
    if not text:
        return None

    matches = list(FINAL_ANSWER_PATTERN.finditer(text))
    if matches:
        return matches[-1].group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    return lines[-1]


def _normalize_text_value(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = [str(part).strip() for part in value if part is not None and str(part).strip()]
        return "\n".join(parts) if parts else None
    if isinstance(value, dict):
        return None
    return str(value)


def _get_first_text_field(example: dict, field_names: Sequence[str]) -> Optional[str]:
    for field_name in field_names:
        if field_name in example:
            normalized = _normalize_text_value(example[field_name])
            if normalized:
                return normalized
    return None


def _select_split(dataset_obj, split_name: str):
    if isinstance(dataset_obj, DatasetDict):
        if split_name in dataset_obj:
            return dataset_obj[split_name]
        if len(dataset_obj) == 1:
            return next(iter(dataset_obj.values()))
        available = ", ".join(dataset_obj.keys())
        raise ValueError(f"Split '{split_name}' not found. Available splits: {available}")
    return dataset_obj


def _to_iterable_dataset(dataset_obj):
    if isinstance(dataset_obj, Dataset):
        return dataset_obj.to_iterable_dataset()
    return dataset_obj


def _infer_split_name(file_path: Path, root_path: Path) -> str:
    relative_parts = [part.lower() for part in file_path.relative_to(root_path).parts]
    candidate_parts = []

    for part in relative_parts:
        candidate_parts.append(Path(part).stem)
        candidate_parts.append(part)

    for part in reversed(candidate_parts):
        for split_name in ("train", "test", "validation", "valid", "val", "dev"):
            if part == split_name:
                if split_name in {"valid", "val"}:
                    return "validation"
                return split_name
            if part.startswith(f"{split_name}-") or part.startswith(f"{split_name}_"):
                if split_name in {"valid", "val"}:
                    return "validation"
                return split_name

    return "train"


def _discover_local_data_files(dataset_path: Path):
    matched_files = {}
    for suffix, format_name in MixedTokenSource.SUPPORTED_SUFFIXES.items():
        files = sorted(
            path
            for path in dataset_path.rglob(f"*{suffix}")
            if path.is_file() and path.name not in MixedTokenSource.IGNORED_JSON_FILES
        )
        if files:
            matched_files[format_name] = files

    if not matched_files:
        raise ValueError(f"No supported dataset files found under {dataset_path}")
    if len(matched_files) > 1:
        formats = ", ".join(sorted(matched_files))
        raise ValueError(f"Found multiple data formats in {dataset_path}: {formats}")

    format_name, files = next(iter(matched_files.items()))
    data_files = {}
    for file_path in files:
        split = _infer_split_name(file_path, dataset_path)
        data_files.setdefault(split, []).append(str(file_path))

    return format_name, data_files


def load_dataset_source(dataset_name: str, split_name: str = "train"):
    """Load either a remote dataset name or a local dataset path."""
    dataset_path = Path(dataset_name).expanduser()
    load_errors = []

    if dataset_path.exists():
        try:
            dataset_obj = load_from_disk(str(dataset_path))
            return _to_iterable_dataset(_select_split(dataset_obj, split_name))
        except Exception as exc:
            load_errors.append(f"load_from_disk failed: {exc}")

        try:
            dataset_obj = load_dataset(str(dataset_path), split=split_name, streaming=True)
            return _to_iterable_dataset(dataset_obj)
        except Exception as exc:
            load_errors.append(f"load_dataset(path, streaming=True) failed: {exc}")

        try:
            dataset_obj = load_dataset(str(dataset_path), split=split_name)
            return _to_iterable_dataset(dataset_obj)
        except Exception as exc:
            load_errors.append(f"load_dataset(path) failed: {exc}")

        try:
            format_name, data_files = _discover_local_data_files(dataset_path)
            dataset_obj = load_dataset(format_name, data_files=data_files, split=split_name)
            return _to_iterable_dataset(dataset_obj)
        except Exception as exc:
            load_errors.append(f"data_files load failed: {exc}")

        errors = "\n".join(f"- {message}" for message in load_errors)
        raise ValueError(f"Could not load local dataset {dataset_name}:\n{errors}")

    try:
        dataset_obj = load_dataset(dataset_name, split=split_name, streaming=True)
        return _to_iterable_dataset(dataset_obj)
    except Exception as exc:
        load_errors.append(f"streaming load failed: {exc}")

    try:
        dataset_obj = load_dataset(dataset_name, split=split_name)
        return _to_iterable_dataset(dataset_obj)
    except Exception as exc:
        load_errors.append(f"standard load failed: {exc}")

    errors = "\n".join(f"- {message}" for message in load_errors)
    raise ValueError(f"Could not load dataset {dataset_name}:\n{errors}")


def extract_question_and_response(example: dict) -> Optional[Tuple[str, str, Optional[str]]]:
    """Extract a question/response pair from a dataset example."""
    question = _get_first_text_field(example, MixedTokenSource.QUERY_FIELDS)
    response = _get_first_text_field(example, MixedTokenSource.SOLUTION_FIELDS)
    combined = _get_first_text_field(example, MixedTokenSource.COMBINED_TEXT_FIELDS)

    if question and response:
        return question, response, extract_final_answer(response)

    if combined is not None:
        solution_match = SOLUTION_HEADER_PATTERN.search(combined)
        if solution_match is not None:
            response = combined[solution_match.end() :].strip()
            question_prefix = combined[: solution_match.start()].strip()
            if not question:
                question = question_prefix
            if response:
                return question or "", response, extract_final_answer(response)

        if question:
            return question, combined, extract_final_answer(combined)

    if question:
        return question, "", extract_final_answer(response or combined)

    return None


def get_input_device(model: AutoModelForCausalLM) -> torch.device:
    """Return the device that should receive tokenizer outputs."""
    if hasattr(model, "get_input_embeddings"):
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is not None:
            return input_embeddings.weight.device
    return next(model.parameters()).device


def get_target_layer(model: AutoModelForCausalLM, hook_layer: int):
    """Locate a transformer block by index."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[hook_layer]
    if hasattr(model, "layers"):
        return model.layers[hook_layer]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[hook_layer]
    raise ValueError("Could not find layers in model architecture.")


def create_reasoning_samples(
    dataset_path: str,
    delimiter: str,
    max_samples: Optional[int] = None,
    response_source: str = "dataset",
    model_name: Optional[str] = None,
    prompt_template: str = "{question}",
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 42,
    response_cache_path: Optional[Path] = None,
    resume_response_cache: bool = False,
    overwrite_response_cache: bool = False,
    prefetched_responses: Optional[Dict[str, Dict]] = None,
) -> List[ReasoningSample]:
    """Load reasoning samples from a dataset, optionally generating fresh responses."""
    dataset_iter = iter(load_dataset_source(dataset_path, split_name="train"))
    samples: List[ReasoningSample] = []
    segmenter = DelimiterStepSegmenter(delimiter)
    progress_desc = (
        "Generating model responses"
        if response_source == "model"
        else "Preparing reasoning samples"
    )
    progress = tqdm(total=max_samples, desc=progress_desc, unit="sample")

    tokenizer = None
    model = None
    input_device = None
    cached_response_records: Dict[str, Dict] = dict(prefetched_responses or {})
    cache_handle = None
    if response_source == "model":
        if not model_name:
            raise ValueError("--model is required when --response-source=model")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        input_device = get_input_device(model)
        if temperature == 0.0:
            torch.manual_seed(seed)
        if response_cache_path is not None:
            response_cache_path.parent.mkdir(parents=True, exist_ok=True)
            file_mode, resume_state = prepare_jsonl_output(
                response_cache_path,
                key_fields=("sample_id",),
                resume=resume_response_cache,
                overwrite=overwrite_response_cache,
            )
            cache_records, skipped_invalid_lines = (
                load_jsonl_records(response_cache_path)
                if file_mode == "a"
                else ([], 0)
            )
            for record in cache_records:
                sample_id = str(record["sample_id"])
                cached_response_records[sample_id] = record
            if response_cache_path:
                print(f"Model response cache: {response_cache_path}")
            if resume_state.loaded_records:
                print(
                    f"Recovered {resume_state.loaded_records} cached model responses "
                    f"from {response_cache_path}."
                )
            if skipped_invalid_lines:
                print(
                    f"Ignored {skipped_invalid_lines} malformed trailing JSONL line(s) "
                    f"while resuming {response_cache_path}."
                )
            cache_handle = response_cache_path.open(file_mode, encoding="utf-8")

    try:
        for raw_index, example in enumerate(dataset_iter):
            parsed = extract_question_and_response(example)
            if parsed is None:
                continue

            question, response, reference_answer = parsed
            if response_source == "model":
                sample_id = str(raw_index)
                cached_record = cached_response_records.get(sample_id)
                if cached_record is not None:
                    cached_question = cached_record.get("question", "")
                    cached_response = cached_record.get("response", "")
                    if cached_question and question and cached_question != question:
                        raise ValueError(
                            f"Cached response for sample_id={sample_id} does not match the current question."
                        )
                    question = cached_question or question
                    response = cached_response
                    reference_answer = cached_record.get("reference_answer", reference_answer)
                else:
                    if not question:
                        continue
                    prompt = build_prompt(question, prompt_template)
                    response = generate_model_response(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        input_device=input_device,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    if cache_handle is not None:
                        cache_record = {
                            "sample_id": sample_id,
                            "question": question,
                            "response": response,
                            "reference_answer": reference_answer,
                            "model_name": model_name,
                            "prompt_template": prompt_template,
                            "max_new_tokens": max_new_tokens,
                            "temperature": temperature,
                            "top_p": top_p,
                        }
                        cache_handle.write(json.dumps(cache_record, ensure_ascii=False) + "\n")
                        cache_handle.flush()
                        cached_response_records[sample_id] = cache_record
            elif not response:
                continue

            full_text = f"Question:\n{question}\n\nResponse:\n{response}"
            response_start_char = len(full_text) - len(response)
            steps = segmenter.segment(full_text, (response_start_char, len(full_text)))
            if not steps:
                continue

            samples.append(
                ReasoningSample(
                    sample_id=str(raw_index),
                    question=question,
                    response=response,
                    full_text=full_text,
                    response_start_char=response_start_char,
                    steps=steps,
                    reference_answer=reference_answer,
                )
            )
            progress.update(1)

            if max_samples is not None and len(samples) >= max_samples:
                break
    finally:
        progress.close()
        if cache_handle is not None:
            cache_handle.close()
        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return samples


def generate_model_response(
    model: AutoModelForCausalLM,
    tokenizer,
    prompt: str,
    input_device: torch.device,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    """Generate a reasoning response from the target model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
    generation = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generation[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def write_step_labels(
    samples: Sequence[ReasoningSample],
    judge: StepJudge,
    output_path: Path,
    judge_name: str,
    judge_model: Optional[str] = None,
    resume: bool = False,
    overwrite_output: bool = False,
) -> Dict[str, int]:
    """Label reasoning steps and write them to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_mode, resume_state = prepare_jsonl_output(
        output_path,
        key_fields=("sample_id", "step_id"),
        resume=resume,
        overwrite=overwrite_output,
    )
    n_written_records = 0
    total_steps = sum(len(sample.steps) for sample in samples)
    progress = tqdm(
        total=total_steps,
        initial=len(resume_state.completed_keys),
        desc=f"Labeling steps ({judge_name})",
        unit="step",
    )

    try:
        with output_path.open(file_mode, encoding="utf-8") as handle:
            for sample in samples:
                for step_index, step in enumerate(sample.steps):
                    step_key = (sample.sample_id, step_index)
                    if step_key in resume_state.completed_keys:
                        continue

                    step_text = sample.full_text[sample.steps[step_index].text_span[0] : sample.steps[step_index].text_span[1]]
                    previous_step = None
                    next_step = None
                    if step_index > 0:
                        previous_span = sample.steps[step_index - 1].text_span
                        previous_step = sample.full_text[previous_span[0] : previous_span[1]]
                    if step_index + 1 < len(sample.steps):
                        next_span = sample.steps[step_index + 1].text_span
                        next_step = sample.full_text[next_span[0] : next_span[1]]

                    try:
                        label, rationale = judge.classify_step(
                            question=sample.question,
                            response=sample.response,
                            step_text=step_text,
                            previous_step=previous_step,
                            next_step=next_step,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "Judge classification failed for "
                            f"sample_id={sample.sample_id}, step_id={step_index}, "
                            f"question_chars={len(sample.question)}, "
                            f"response_chars={len(sample.response)}, "
                            f"step_chars={len(step_text)}"
                        ) from exc
                    record = {
                        "sample_id": sample.sample_id,
                        "question": sample.question,
                        "response": sample.response,
                        "full_text": sample.full_text,
                        "reference_answer": sample.reference_answer,
                        "step_id": step_index,
                        "step_text": step_text,
                        "step_start_char": step.text_span[0],
                        "step_end_char": step.text_span[1],
                        "delimiter_start_char": step.delimiter_span[0] if step.delimiter_span else None,
                        "delimiter_end_char": step.delimiter_span[1] if step.delimiter_span else None,
                        "label": label,
                        "label_rationale": rationale,
                        "label_source": judge_name,
                        "judge_model": judge_model,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    n_written_records += 1
                    progress.update(1)
    finally:
        progress.close()

    return {
        "written_records": n_written_records,
        "recovered_records": resume_state.loaded_records,
        "skipped_invalid_lines": resume_state.skipped_invalid_lines,
        "total_records": len(resume_state.completed_keys) + n_written_records,
    }


def load_step_records(path: Path) -> List[Dict]:
    """Load JSONL step records."""
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _group_records_by_sample(records: Sequence[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        grouped[record["sample_id"]].append(record)
    for sample_records in grouped.values():
        sample_records.sort(key=lambda item: item["step_id"])
    return grouped


def _make_step_from_record(record: Dict) -> ReasoningStep:
    delimiter_span = None
    if record.get("delimiter_start_char") is not None and record.get("delimiter_end_char") is not None:
        delimiter_span = (record["delimiter_start_char"], record["delimiter_end_char"])
    return ReasoningStep(
        text_span=(record["step_start_char"], record["step_end_char"]),
        delimiter_span=delimiter_span,
    )


def capture_step_activations(
    records: Sequence[Dict],
    model_name: str,
    hook_layer: int,
    max_seq_length: int = 4096,
) -> List[Tuple[Dict, torch.Tensor]]:
    """Capture step-level activations for labeled records."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    input_device = get_input_device(model)
    target_layer = get_target_layer(model, hook_layer)
    selector = ActivationPointSelector("step_delimiter")
    grouped = _group_records_by_sample(records)
    outputs: List[Tuple[Dict, torch.Tensor]] = []
    captured = {}

    def hook_fn(module, inputs, output):
        captured["activations"] = inputs[0].detach() if isinstance(inputs, tuple) else inputs.detach()

    hook_handle = target_layer.register_forward_hook(hook_fn)
    try:
        with torch.inference_mode():
            for sample_records in tqdm(grouped.values(), desc="Capturing step activations"):
                full_text = sample_records[0]["full_text"]
                encoded = tokenizer(
                    full_text,
                    max_length=max_seq_length,
                    truncation=True,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                )
                if "offset_mapping" not in encoded:
                    raise ValueError(
                        "Tokenizer must support offset mappings for paper-style step evaluation."
                    )
                offsets = encoded["offset_mapping"].squeeze(0)
                tokenized = {key: value.to(input_device) for key, value in encoded.items() if key != "offset_mapping"}
                steps = [_make_step_from_record(record) for record in sample_records]
                token_indices = selector.select_indices(offsets, steps)

                captured.clear()
                model(**tokenized, use_cache=False)
                hidden_states = captured.get("activations")
                if hidden_states is None:
                    continue

                for record, token_index in zip(sample_records, token_indices):
                    if token_index is None or token_index >= hidden_states.shape[1]:
                        continue
                    outputs.append((record, hidden_states[0, token_index, :].detach().cpu()))
    finally:
        hook_handle.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return outputs


def score_behavior_features(
    step_activations: Sequence[Tuple[Dict, torch.Tensor]],
    sae,
    top_features_per_label: int = 20,
) -> Dict:
    """Score SAE features by their behavior specificity."""
    labels = sorted({record["label"] for record, _ in step_activations})
    if not labels:
        raise ValueError("No labeled step activations were available.")

    device = next(sae.parameters()).device
    label_sums = {
        label: torch.zeros(sae.d_sae, device=device, dtype=sae.W_dec.dtype)
        for label in labels
    }
    label_counts = Counter()

    with torch.inference_mode():
        for record, activation in tqdm(step_activations, desc="Scoring SAE features"):
            batch = activation.to(device=device, dtype=sae.W_enc.dtype).unsqueeze(0)
            _, features, _, _ = sae(batch)
            label_sums[record["label"]] += features.squeeze(0)
            label_counts[record["label"]] += 1

    label_means = {}
    for label in labels:
        if label_counts[label] == 0:
            label_means[label] = torch.zeros_like(label_sums[label])
        else:
            label_means[label] = label_sums[label] / label_counts[label]

    behavior_stats = {}
    feature_assignments = {}
    for label in labels:
        if len(labels) > 1:
            other_means = torch.stack([label_means[other] for other in labels if other != label], dim=0)
            other_max = other_means.max(dim=0).values
        else:
            other_max = torch.zeros_like(label_means[label])
        specificity = label_means[label] - other_max
        top_values, top_indices = torch.topk(
            specificity,
            min(top_features_per_label, specificity.shape[0]),
        )
        top_features = []
        for feature_index, score_value in zip(top_indices.tolist(), top_values.tolist()):
            mean_activation = float(label_means[label][feature_index].item())
            entry = {
                "feature_index": feature_index,
                "specificity": score_value,
                "mean_activation": mean_activation,
            }
            top_features.append(entry)
            previous = feature_assignments.get(feature_index)
            if previous is None or score_value > previous["specificity"]:
                feature_assignments[feature_index] = {
                    "assigned_label": label,
                    "specificity": score_value,
                    "mean_activation": mean_activation,
                }

        behavior_stats[label] = {
            "num_steps": label_counts[label],
            "top_features": top_features,
        }

    return {
        "labels": labels,
        "behavior_stats": behavior_stats,
        "feature_assignments": feature_assignments,
    }


def embed_decoder_features(
    sae,
    feature_assignments: Dict[int, Dict],
    method: str = "umap",
    random_seed: int = 42,
) -> List[Dict]:
    """Project selected decoder rows into a 2-D embedding space."""
    feature_indices = sorted(feature_assignments)
    if not feature_indices:
        return []

    rows = sae.W_dec[feature_indices].detach().cpu().float()
    rows = F.normalize(rows, dim=1)

    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "UMAP embedding requires the optional dependency 'umap-learn'."
            ) from exc
        coords = umap.UMAP(metric="cosine", random_state=random_seed).fit_transform(rows.numpy())
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
    elif method == "pca":
        coords_tensor = pca_project(rows)
    else:
        raise ValueError(f"Unsupported embedding method: {method}")

    points = []
    for feature_index, coord in zip(feature_indices, coords_tensor.tolist()):
        assignment = feature_assignments[feature_index]
        points.append(
            {
                "feature_index": feature_index,
                "assigned_label": assignment["assigned_label"],
                "specificity": assignment["specificity"],
                "mean_activation": assignment["mean_activation"],
                "x": coord[0],
                "y": coord[1],
            }
        )
    return points


def pca_project(rows: torch.Tensor) -> torch.Tensor:
    """Project decoder rows to 2-D using PCA."""
    centered = rows - rows.mean(dim=0, keepdim=True)
    q = min(2, centered.shape[1], centered.shape[0])
    _, _, v = torch.pca_lowrank(centered, q=q)
    projected = centered @ v[:, :q]
    if projected.shape[1] == 1:
        projected = torch.cat([projected, torch.zeros_like(projected)], dim=1)
    return projected[:, :2]


def compute_silhouette(points: Sequence[Dict]) -> Optional[float]:
    """Compute a cosine-distance silhouette score for embedded decoder features."""
    if len(points) < 3:
        return None

    labels = [point["assigned_label"] for point in points]
    if len(set(labels)) < 2:
        return None

    coords = torch.tensor([[point["x"], point["y"]] for point in points], dtype=torch.float32)
    coords = F.normalize(coords, dim=1)
    distances = 1 - coords @ coords.T

    scores = []
    for index, label in enumerate(labels):
        same_indices = [i for i, other in enumerate(labels) if other == label and i != index]
        other_labels = sorted(set(labels) - {label})
        if not same_indices or not other_labels:
            continue

        a = distances[index, same_indices].mean().item()
        b = min(
            distances[index, [i for i, other in enumerate(labels) if other == other_label]].mean().item()
            for other_label in other_labels
        )
        denominator = max(a, b)
        if denominator > 0:
            scores.append((b - a) / denominator)

    if not scores:
        return None
    return float(sum(scores) / len(scores))


def save_geometry_outputs(
    output_dir: Path,
    summary: Dict,
    points: Sequence[Dict],
    plot_path: Optional[Path] = None,
):
    """Persist geometry summaries and optional scatter plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "geometry_summary.json"
    points_path = output_dir / "decoder_points.jsonl"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    with points_path.open("w", encoding="utf-8") as handle:
        for point in points:
            handle.write(json.dumps(point, ensure_ascii=False) + "\n")

    if plot_path is not None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "Plotting decoder geometry requires the optional dependency 'matplotlib'."
            ) from exc

        label_colors = {
            "reflection": "#c43c39",
            "backtracking": "#2b6cb0",
            "other": "#7a7a7a",
        }
        plt.figure(figsize=(8, 6))
        for label in sorted({point["assigned_label"] for point in points}):
            label_points = [point for point in points if point["assigned_label"] == label]
            plt.scatter(
                [point["x"] for point in label_points],
                [point["y"] for point in label_points],
                label=label,
                alpha=0.75,
                s=28,
                c=label_colors.get(label, "#444444"),
            )
        plt.title("Decoder Geometry by Behavior Label")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend()
        plt.tight_layout()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=180)
        plt.close()


def load_geometry_summary(summary_path: Path) -> Dict:
    """Load a geometry summary JSON file."""
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_behavior_vector(
    sae,
    summary: Dict,
    behavior: str,
    top_n_features: int = 8,
) -> torch.Tensor:
    """Average top behavior-specific decoder rows into a steering vector."""
    features = summary["behavior_stats"].get(behavior, {}).get("top_features", [])
    if not features:
        raise ValueError(f"No features found for behavior '{behavior}'.")

    feature_indices = [item["feature_index"] for item in features[:top_n_features]]
    rows = sae.W_dec[feature_indices].detach().float()
    rows = F.normalize(rows, dim=1)
    vector = rows.mean(dim=0)
    return F.normalize(vector, dim=0)


def parse_condition_spec(spec: str) -> List[Tuple[str, float]]:
    """Parse a comma-separated intervention condition spec."""
    conditions = []
    for chunk in spec.split(","):
        name, value = chunk.split(":", 1)
        conditions.append((name.strip(), float(value.strip())))
    return conditions


def sequence_ends_with(sequence: torch.Tensor, suffix: Sequence[int]) -> bool:
    """Check whether a token sequence ends with a given suffix."""
    if not suffix or sequence.shape[0] < len(suffix):
        return False
    return sequence[-len(suffix) :].tolist() == list(suffix)


def _generate_with_loaded_model(
    model,
    tokenizer,
    input_device: torch.device,
    target_layer,
    prompt: str,
    behavior_vector: torch.Tensor,
    alpha: float,
    delimiter: str = "\n\n",
    max_new_tokens: int = 384,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 42,
) -> str:
    """Generate a response while injecting a behavior vector at step boundaries."""
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    delimiter_ids = tokenizer.encode(delimiter, add_special_tokens=False)
    vector = behavior_vector.to(input_device, dtype=torch.bfloat16)
    generation_state = {"apply": False}

    def pre_hook(module, inputs):
        if not generation_state["apply"] or alpha == 0.0:
            return None
        hidden_states = inputs[0]
        last_hidden = hidden_states[:, -1, :]
        projection = (last_hidden * vector).sum(dim=-1, keepdim=True)
        updated = hidden_states.clone()
        updated[:, -1, :] = last_hidden - alpha * projection * vector
        return (updated,) + inputs[1:]

    hook_handle = target_layer.register_forward_pre_hook(pre_hook)

    try:
        if temperature == 0.0:
            torch.manual_seed(seed)

        input_ids = prompt_ids.to(input_device)
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                generation_state["apply"] = sequence_ends_with(input_ids[0], delimiter_ids)
                outputs = model(input_ids=input_ids, use_cache=False)
                next_token_logits = outputs.logits[:, -1, :]

                if temperature > 0.0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    if top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative = torch.cumsum(sorted_probs, dim=-1)
                        cutoff = cumulative > top_p
                        cutoff[..., 1:] = cutoff[..., :-1].clone()
                        cutoff[..., 0] = False
                        sorted_probs[cutoff] = 0.0
                        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                        next_token = sorted_indices.gather(
                            -1,
                            torch.multinomial(sorted_probs, num_samples=1),
                        )
                    else:
                        next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                    break

        generated = tokenizer.decode(
            input_ids[0][prompt_ids.shape[1] :],
            skip_special_tokens=True,
        )
        return generated.strip()
    finally:
        hook_handle.remove()


def generate_with_intervention(
    model_name: str,
    prompt: str,
    hook_layer: int,
    behavior_vector: torch.Tensor,
    alpha: float,
    delimiter: str = "\n\n",
    max_new_tokens: int = 384,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 42,
) -> str:
    """Generate a response while injecting a behavior vector at step boundaries."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    input_device = get_input_device(model)
    target_layer = get_target_layer(model, hook_layer)

    try:
        return _generate_with_loaded_model(
            model=model,
            tokenizer=tokenizer,
            input_device=input_device,
            target_layer=target_layer,
            prompt=prompt,
            behavior_vector=behavior_vector,
            alpha=alpha,
            delimiter=delimiter,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def label_generated_steps(
    question: str,
    response: str,
    delimiter: str,
    judge: StepJudge,
) -> Counter:
    """Label steps from a generated response and return behavior counts."""
    full_text = f"Question:\n{question}\n\nResponse:\n{response}"
    response_start = len(full_text) - len(response)
    steps = DelimiterStepSegmenter(delimiter).segment(full_text, (response_start, len(full_text)))
    counts = Counter()

    for index, step in enumerate(steps):
        step_text = full_text[step.text_span[0] : step.text_span[1]]
        previous_step = None
        next_step = None
        if index > 0:
            previous_span = steps[index - 1].text_span
            previous_step = full_text[previous_span[0] : previous_span[1]]
        if index + 1 < len(steps):
            next_span = steps[index + 1].text_span
            next_step = full_text[next_span[0] : next_span[1]]

        try:
            label, _ = judge.classify_step(
                question=question,
                response=response,
                step_text=step_text,
                previous_step=previous_step,
                next_step=next_step,
            )
        except Exception as exc:
            raise RuntimeError(
                "Judge classification failed for generated response "
                f"step_id={index}, question_chars={len(question)}, "
                f"response_chars={len(response)}, step_chars={len(step_text)}"
            ) from exc
        counts[label] += 1

    return counts


def run_intervention_eval(
    samples: Sequence[ReasoningSample],
    model_name: str,
    hook_layer: int,
    behavior_vector: torch.Tensor,
    conditions: Sequence[Tuple[str, float]],
    judge: StepJudge,
    delimiter: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    output_path: Path,
    seed: int = 42,
    resume: bool = False,
    overwrite_output: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Run causal interventions and save per-sample results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_mode, resume_state = prepare_jsonl_output(
        output_path,
        key_fields=("sample_id", "condition"),
        resume=resume,
        overwrite=overwrite_output,
    )
    aggregates = defaultdict(
        lambda: {
            "reflection": 0,
            "backtracking": 0,
            "other": 0,
            "correct": 0,
            "scored": 0,
            "total": 0,
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    input_device = get_input_device(model)
    target_layer = get_target_layer(model, hook_layer)
    recovered_records, _ = load_jsonl_records(output_path) if resume else ([], 0)
    for record in recovered_records:
        condition_name = record["condition"]
        aggregate = aggregates[condition_name]
        aggregate["reflection"] += record.get("reflection_steps", 0)
        aggregate["backtracking"] += record.get("backtracking_steps", 0)
        aggregate["other"] += record.get("other_steps", 0)
        aggregate["total"] += 1
        is_correct = record.get("is_correct")
        if is_correct is not None:
            aggregate["scored"] += 1
            if is_correct:
                aggregate["correct"] += 1
    progress = tqdm(
        total=len(samples) * len(conditions),
        initial=len(resume_state.completed_keys),
        desc="Running interventions",
        unit="run",
    )

    try:
        with output_path.open(file_mode, encoding="utf-8") as handle:
            for sample in samples:
                prompt = sample.question
                for condition_name, alpha in conditions:
                    record_key = (sample.sample_id, condition_name)
                    if record_key in resume_state.completed_keys:
                        progress.update(1)
                        continue

                    response = _generate_with_loaded_model(
                        model=model,
                        tokenizer=tokenizer,
                        input_device=input_device,
                        target_layer=target_layer,
                        prompt=prompt,
                        behavior_vector=behavior_vector,
                        alpha=alpha,
                        delimiter=delimiter,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed,
                    )
                    counts = label_generated_steps(sample.question, response, delimiter, judge)
                    predicted_answer = extract_final_answer(response)
                    is_correct = None
                    if sample.reference_answer:
                        is_correct = normalize_answer(predicted_answer) == normalize_answer(
                            sample.reference_answer
                        )

                    record = {
                        "sample_id": sample.sample_id,
                        "condition": condition_name,
                        "alpha": alpha,
                        "question": sample.question,
                        "reference_answer": sample.reference_answer,
                        "predicted_answer": predicted_answer,
                        "is_correct": is_correct,
                        "reflection_steps": counts.get("reflection", 0),
                        "backtracking_steps": counts.get("backtracking", 0),
                        "other_steps": counts.get("other", 0),
                        "response": response,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    progress.update(1)

                    aggregate = aggregates[condition_name]
                    aggregate["reflection"] += record["reflection_steps"]
                    aggregate["backtracking"] += record["backtracking_steps"]
                    aggregate["other"] += record["other_steps"]
                    aggregate["total"] += 1
                    if is_correct is not None:
                        aggregate["scored"] += 1
                        if is_correct:
                            aggregate["correct"] += 1
    finally:
        progress.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {}
    for condition_name, stats in aggregates.items():
        total = max(stats["total"], 1)
        accuracy = None
        if stats["scored"] > 0:
            accuracy = stats["correct"] / stats["scored"]
        summary[condition_name] = {
            "mean_reflection_steps": stats["reflection"] / total,
            "mean_backtracking_steps": stats["backtracking"] / total,
            "mean_other_steps": stats["other"] / total,
            "accuracy": accuracy,
            "num_samples": stats["total"],
            "num_scored_samples": stats["scored"],
        }
    return summary
