"""Tools for data enrichment.

This module contains functions that are directly exposed to the LLM as tools.
These tools can be used for tasks such as web searching and scraping.
Users can edit and extend these tools as needed.
"""

from typing import Any, Optional, cast

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

from enrichment_agent.configuration import Configuration


async def search_tavily(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Query a search engine.

    This function queries the web to fetch comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events. Provide as much context in the query as needed to ensure high recall.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_tavily_search_results)
    result = await wrapped.ainvoke({"query": query})

    return cast(list[dict[str, Any]], result)


async def search_wikipedia(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Query a Wikipedia engine.

    This function queries the Wikipedia to fetch comprehensive, accurate, and trusted results
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = WikipediaLoader(
        query=query, load_max_docs=configuration.max_wikipedia_search_results
    )
    result = await wrapped.aload()

    return cast(list[dict[str, Any]], result)
