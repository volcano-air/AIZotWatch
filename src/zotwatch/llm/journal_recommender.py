"""Target journal list generation service using LLM."""

import json
import logging
import re
from dataclasses import dataclass

from zotwatch.core.models import VenueStats

from .base import BaseLLMProvider
from .prompts import JOURNAL_GENERATION_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class GeneratedJournal:
    """A journal proposed by the LLM, before ISSN verification.

    The ISSN is intentionally not requested from the LLM; it is resolved
    later via Crossref to guarantee authoritative values.
    """

    title: str
    category: str
    impact_factor: float | None = None
    is_chinese: bool = False


class JournalRecommender:
    """Generates a target journal list from library venues using an LLM."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        model: str | None = None,
    ):
        """Initialize the recommender.

        Args:
            llm: LLM provider instance.
            model: Optional model name to use.
        """
        self.llm = llm
        self.model = model

    def generate(
        self,
        venues: list[VenueStats],
        research_focus: str = "",
        max_tokens: int = 4096,
    ) -> list[GeneratedJournal]:
        """Generate candidate journals from the library's top venues.

        Args:
            venues: Venue statistics extracted from the user's library.
            research_focus: Optional description to steer recommendations.
            max_tokens: Token budget for the completion (journal lists can be long).

        Returns:
            List of proposed journals (without ISSN).
        """
        if not venues:
            return []

        venues_list = "\n".join(
            f"- {v.venue} [{v.venue_type}]: {v.paper_count} 篇" for v in venues
        )
        prompt = JOURNAL_GENERATION_PROMPT.format(
            venues_list=venues_list,
            research_focus=research_focus or "未提供",
        )

        response = self.llm.complete(prompt, model=self.model, max_tokens=max_tokens)
        logger.debug("LLM response for journal generation: %s", response.content)

        return self._parse_response(response.content)

    def _parse_response(self, content: str | None) -> list[GeneratedJournal]:
        """Parse LLM JSON response into a list of GeneratedJournal."""
        if content is None:
            logger.warning("LLM returned None content for journal generation")
            return []

        try:
            content = content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n?", "", content)
                content = re.sub(r"\n?```$", "", content)

            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse journal generation response: %s", e)
            return []

        results: list[GeneratedJournal] = []
        for entry in data.get("journals", []):
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            raw_if = entry.get("impact_factor")
            try:
                impact_factor = float(raw_if) if raw_if is not None else None
            except (TypeError, ValueError):
                impact_factor = None
            results.append(
                GeneratedJournal(
                    title=title,
                    category=(entry.get("category") or "").strip(),
                    impact_factor=impact_factor,
                    is_chinese=bool(entry.get("is_chinese", False)),
                )
            )

        return results


__all__ = ["JournalRecommender", "GeneratedJournal"]
