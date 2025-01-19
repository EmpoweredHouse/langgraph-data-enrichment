"""State definitions.

State is the interface between the graph and end user as well as the
data model used internally by the graph.
"""

import operator
from dataclasses import dataclass, field
from typing import Annotated, List, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from sqlalchemy import false


class Analyst(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives."
    )

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )


@dataclass(kw_only=True)
class GenerateAnalystsState:
    topic: str
    """Research topic"""

    max_analysts: int
    """Number of analysts"""

    human_analyst_feedback: Optional[str] = field(default=None)
    """Human feedback"""

    analysts: List[Analyst] = field(default_factory=list)
    """Analyst asking questions"""


@dataclass(kw_only=True)
class InterviewState:
    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)

    max_num_turns: int = field(default=2)
    """Number turns of conversation"""

    context: Annotated[list, operator.add]
    """Source docs"""

    analyst: Analyst
    """Analyst asking questions"""

    interview: str = field(default=None)
    """Interview transcript"""

    sections: list = field(default_factory=list)
    """Final key we duplicate in outer state for Send() API"""


@dataclass(kw_only=True)
class ResearchGraphState:
    topic: str
    """Research topic"""

    max_analysts: int
    """Number of analysts"""

    human_analyst_feedback: str
    """Human feedback"""

    analysts: List[Analyst]
    """Analyst asking questions"""

    sections: Annotated[list, operator.add] = field(default_factory=list)
    """Send() API key"""

    introduction: str = ""
    """Introduction for the final report"""

    content: str = ""
    """Content for the final report"""

    conclusion: str = ""
    """Conclusion for the final report"""

    final_report: str = ""
    """Final report"""
