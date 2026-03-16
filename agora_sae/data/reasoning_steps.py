"""Utilities for reasoning step segmentation and activation-point selection."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch


CharSpan = Tuple[int, int]


@dataclass(frozen=True)
class ReasoningStep:
    """A reasoning step plus the delimiter that follows it, if present."""

    text_span: CharSpan
    delimiter_span: Optional[CharSpan] = None


class DelimiterStepSegmenter:
    """Split a reasoning region into steps using a delimiter string."""

    def __init__(self, delimiter: Optional[str] = "\n\n"):
        self.delimiter = delimiter or "\n\n"

    @staticmethod
    def _trim_span(text: str, start: int, end: int) -> Optional[CharSpan]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start >= end:
            return None
        return start, end

    def segment(self, text: str, span: CharSpan) -> List[ReasoningStep]:
        """Return trimmed step spans within a reasoning region."""
        start, end = span
        if start >= end:
            return []

        steps: List[ReasoningStep] = []
        cursor = start

        while cursor < end:
            delimiter_index = text.find(self.delimiter, cursor, end)
            if delimiter_index == -1:
                step_end = end
                delimiter_span = None
                next_cursor = end
            else:
                step_end = delimiter_index
                delimiter_span = (delimiter_index, delimiter_index + len(self.delimiter))
                next_cursor = delimiter_span[1]

            trimmed_step = self._trim_span(text, cursor, step_end)
            if trimmed_step is not None:
                steps.append(ReasoningStep(text_span=trimmed_step, delimiter_span=delimiter_span))

            cursor = next_cursor

        return steps


class ActivationPointSelector:
    """Map reasoning steps to token positions used for activation extraction."""

    def __init__(self, strategy: str = "step_delimiter"):
        if strategy != "step_delimiter":
            raise ValueError(f"Unsupported activation point strategy: {strategy}")
        self.strategy = strategy

    def select_mask(
        self,
        offsets: torch.Tensor,
        steps: Sequence[ReasoningStep],
    ) -> torch.Tensor:
        """Build a token mask for the selected activation points."""
        mask = torch.zeros(offsets.shape[0], dtype=torch.bool)

        for selected_index in self.select_indices(offsets, steps):
            if selected_index is not None:
                mask[selected_index] = True

        return mask

    def select_indices(
        self,
        offsets: torch.Tensor,
        steps: Sequence[ReasoningStep],
    ) -> List[Optional[int]]:
        """Return the token index selected for each reasoning step."""
        return [self._select_index_for_step(offsets, step) for step in steps]

    def _select_index_for_step(
        self,
        offsets: torch.Tensor,
        step: ReasoningStep,
    ) -> Optional[int]:
        if step.delimiter_span is not None:
            delimiter_index = self._last_overlapping_token(offsets, step.delimiter_span)
            if delimiter_index is not None:
                return delimiter_index

        # Fall back to the last token inside the step so single-step
        # solutions still contribute one activation point.
        return self._last_overlapping_token(offsets, step.text_span)

    @staticmethod
    def _last_overlapping_token(
        offsets: torch.Tensor,
        span: CharSpan,
    ) -> Optional[int]:
        start_char, end_char = span
        overlap = (offsets[:, 0] < end_char) & (offsets[:, 1] > start_char)
        token_indices = torch.nonzero(overlap, as_tuple=False).flatten()
        if token_indices.numel() == 0:
            return None
        return int(token_indices[-1].item())
