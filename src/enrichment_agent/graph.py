"""Define a data enrichment agent.

Works with a chat model with tool calling support.
"""

import operator
from typing import Any, Dict, List, Literal, Optional, Union, cast

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from enrichment_agent import prompts
from enrichment_agent.configuration import Configuration
from enrichment_agent.state import (
    GenerateAnalystsState,
    InterviewState,
    Perspectives,
    ResearchGraphState,
)
from enrichment_agent.tools import search_tavily, search_wikipedia
from enrichment_agent.utils import *


async def create_analysts(
    state: GenerateAnalystsState, config: RunnableConfig
) -> Dict[str, Any]:
    """Create analysts"""

    # Load model
    configuration = Configuration.from_runnable_config(config)
    raw_model = init_model(configuration.model)

    structured_llm = raw_model.with_structured_output(Perspectives)

    # Generate question
    analysts_template = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.ANALYST_INSTRUCTIONS_PROMPT),
            ("human", "Generate the set of analysts"),
        ]
    )

    analysts_template_messages = analysts_template.format_messages(
        topic=state.topic,
        human_analyst_feedback=state.human_analyst_feedback or "",
        max_analysts=state.max_analysts,
    )
    analysts = cast(AIMessage, await structured_llm.ainvoke(analysts_template_messages))

    # Write the list of analysis to state
    return {"analysts": analysts.analysts}


def human_feedback(state: GenerateAnalystsState):
    """No-op node that should be interrupted on"""
    return {"human_analyst_feedback": None}


async def generate_question(
    state: InterviewState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Node to generate a question"""

    # Get state
    analyst = state.analyst
    messages = state.messages

    # Load model
    configuration = Configuration.from_runnable_config(config)
    raw_model = init_model(configuration.model)

    # Generate question
    system_message = SystemMessage(
        content=prompts.QUESTION_INSTRUCTIONS_PROMPT.format(goals=analyst.persona)
    )
    question = await raw_model.ainvoke([system_message] + messages)

    # Write messages to state
    return {"messages": [question]}


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")


async def construct_search_query(
    state: InterviewState, *, config: Optional[RunnableConfig] = None
) -> SearchQuery:
    # Load model
    configuration = Configuration.from_runnable_config(config)
    raw_model = init_model(configuration.model)

    # Search query
    structured_llm = raw_model.with_structured_output(SearchQuery)

    search_instructions = SystemMessage(content=prompts.SEARCH_INSTRUCTIONS_PROMPT)
    print(type(state))
    print(state)
    return await structured_llm.ainvoke([search_instructions] + state.messages)


async def search_web(
    state: InterviewState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Retrieve docs from web search"""

    # Search
    search_query = await construct_search_query(state=state, config=config)
    search_docs = await search_tavily(search_query.search_query, config=config)

    # Format
    return {"context": [formatted_web_search_docs(search_docs)]}


async def search_wiki(
    state: InterviewState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Retrieve docs from wikipedia"""
    # Search
    search_query = await construct_search_query(state=state, config=config)
    search_docs = await search_wikipedia(search_query.search_query, config=config)

    return {"context": [formatted_wiki_search_docs(search_docs)]}


def generate_answer(
    state: InterviewState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Node to answer a question"""

    # Load model
    configuration = Configuration.from_runnable_config(config)
    raw_model = init_model(configuration.model)

    # Answer question
    system_message = prompts.ANSWER_INSTRUCTIONS_PROMPT.format(
        goals=state.analyst.persona, context=state.context
    )
    answer = raw_model.invoke([SystemMessage(content=system_message)] + state.messages)

    # Name the message as coming from the expert
    answer.name = "expert"

    # Append it to state
    return {"messages": [answer]}


def save_interview(state: InterviewState) -> Dict[str, Any]:
    """Save interviews"""

    # Convert interview to a string
    interview = get_buffer_string(state.messages)

    # Save to interviews key
    return {"interview": interview}


def route_messages(
    state: InterviewState, name: str = "expert"
) -> Literal["save_interview", "ask_question"]:
    """Route between question and answer"""

    # Get messages
    messages = state.messages
    max_num_turns = state.max_num_turns  # .get('max_num_turns', 2)

    # Check the number of expert answers
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return "save_interview"

    # This router is run after each question - answer pair
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]

    if "Thank you so much for your help" in last_question.content:
        return "save_interview"

    return "ask_question"


def write_section(
    state: InterviewState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Node to write a section"""

    # Get state
    interview = state.interview
    context = state.context
    analyst = state.analyst

    # Load model
    configuration = Configuration.from_runnable_config(config)
    raw_model = init_model(configuration.model)

    # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
    system_message = prompts.SECTION_WRITER_INSTRUCTIONS.format(
        focus=analyst.description
    )
    section = raw_model.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content=f"Use this source to write your section: {context}")]
    )

    # Append it to state
    return {"sections": [section.content]}


# def initiate_all_interviews(state: ResearchGraphState):
def initiate_all_interviews(state: GenerateAnalystsState) -> Union[str, List[Send]]:
    """Conditional edge to initiate all interviews via Send() API or return to create_analysts"""

    if state.human_analyst_feedback:
        # Return to create_analysts
        return "create_analysts"

    # Otherwise kick off interviews in parallel via Send() API
    else:
        topic = state.topic
        return [
            Send(
                "conduct_interview",
                {
                    "analyst": analyst,
                    "messages": [
                        HumanMessage(
                            content=f"So you said you were writing an article on {topic}?"
                        )
                    ],
                },
            )
            for analyst in state.analysts
        ]


def write_report(
    state: ResearchGraphState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Node to write the final report body"""

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in state.sections])

    # Load model
    configuration = Configuration.from_runnable_config(config)
    raw_model = init_model(configuration.model)

    # Summarize the sections into a final report
    system_message = prompts.REPORT_WRITER_INSTRUCTIONS.format(
        topic=state.topic, context=formatted_str_sections
    )
    report = raw_model.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content=f"Write a report based upon these memos.")]
    )

    return {"content": report.content}


def write_introduction(
    state: ResearchGraphState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Node to write the introduction"""

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in state.sections])

    # Summarize the sections into a final report

    # Load model
    configuration = Configuration.from_runnable_config(config)
    raw_model = init_model(configuration.model)

    instructions = prompts.INTRO_CONCLUSTION_INSTRUCTIONS_PROMPT.format(
        topic=state.topic, formatted_str_sections=formatted_str_sections
    )
    intro = raw_model.invoke(
        [instructions] + [HumanMessage(content=f"Write the report introduction")]
    )
    return {"introduction": intro.content}


def write_conclusion(
    state: ResearchGraphState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Node to write the conclusion"""

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in state.sections])

    # Load model
    configuration = Configuration.from_runnable_config(config)
    raw_model = init_model(configuration.model)

    # Summarize the sections into a final report
    instructions = prompts.INTRO_CONCLUSTION_INSTRUCTIONS_PROMPT.format(
        topic=state.topic, formatted_str_sections=formatted_str_sections
    )
    conclusion = raw_model.invoke(
        [instructions] + [HumanMessage(content=f"Write the report conclusion")]
    )
    return {"conclusion": conclusion.content}


def finalize_report(state: ResearchGraphState):
    """The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion"""

    # Save full final report
    content = state.content
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = (
        state.introduction + "\n\n---\n\n" + content + "\n\n---\n\n" + state.conclusion
    )
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}


# Add nodes and edges
interview_builder = StateGraph(InterviewState, config_schema=Configuration)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wiki)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

# Flow
interview_builder.add_edge("__start__", "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges(
    "answer_question", route_messages, ["ask_question", "save_interview"]
)
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", "__end__")

# Add nodes and edges
builder = StateGraph(ResearchGraphState, config_schema=Configuration)

builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", interview_builder.compile())
builder.add_node("write_report", write_report)
builder.add_node("write_introduction", write_introduction)
builder.add_node("write_conclusion", write_conclusion)
builder.add_node("finalize_report", finalize_report)

# Logic
builder.add_edge("__start__", "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges(
    "human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"]
)
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")
builder.add_edge(
    ["write_conclusion", "write_report", "write_introduction"], "finalize_report"
)
builder.add_edge("finalize_report", "__end__")

# Compile
graph = builder.compile(interrupt_before=["human_feedback"])
