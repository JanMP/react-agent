from langchain_openai import ChatOpenAI

import operator
from typing import Annotated, Any
from typing_extensions import TypedDict
from pydantic import BaseModel

from langgraph.graph import END, StateGraph, START
from langgraph.constants import Send

import os
from sqlalchemy import create_engine
import pandas as pd

model = ChatOpenAI(model="gpt-4.1", temperature=0)

from utilities.meteor_client_connection import MeteorClientConnection
connection = MeteorClientConnection("REVIEW_SUMMARY_AGENT")

async def save_to_meteor(data: dict):
    """
    Save the data to the Meteor database.
    """
    await connection.call("thomannNcArt.addReviewSummary", [data])

class Dimension(BaseModel):
    keyword: str
    value: int

class Summary(BaseModel):
    text: str
    topics: list[Dimension]

class OverallState(TypedDict):
    art_artid: str
    reviews: list[dict]
    df: Any  # Use Any instead of pd.DataFrame
    sample_summaries: Annotated[list[Summary], operator.add]
    overall_summary: Summary

class SampleState(BaseModel):
    sample: Any
    sample_size: int
    percentage: float


def prepare_data(state: OverallState):
    
    reviews = state['reviews']
    
    df = pd.DataFrame(reviews)
    df['full_text'] = df['ubi_headl'] + '\n\n ' + df['ubi_ltext']

    state['df'] = df
    return state


def continue_to_sample_summaries(state: OverallState):
    df = state['df']

    # We create samples for good, mid and bad reviews
    bad_filter = df['ubi_bwges'] < 4
    mid_filter = (df['ubi_bwges'] >= 4) & (df['ubi_bwges'] <= 8)
    good_filter = df['ubi_bwges'] > 8

    filters = [bad_filter, mid_filter, good_filter]

    target_sample_size = 100
    random_state = 42

    def get_sample_for_filter(filter) -> SampleState:
        reviews = df[filter]
        percentage = len(reviews) / len(df)
        sample_size = min(len(reviews), target_sample_size)
        sample = reviews.sample(sample_size, random_state=random_state)
        return SampleState(sample=sample, sample_size=sample_size, percentage=percentage)

    # Return a list of Send objects, one for each filter
    return [Send("summarize_sample", get_sample_for_filter(filter)) for filter in filters]


def summarize_sample(state: SampleState):
    sample = state.sample
    sample_size = state.sample_size
    percentage = state.percentage

    if sample_size == 0:
        return None
    
    reviews_text = ''.join([f"Review {i+1}:\n{text}\n\n" for i, text in enumerate(sample['full_text'].tolist())])

    prompt = f"""
        You are a review summarizer. Your task is to summarize a sample of reviews.
        Do the following tasks:
            1. Identify a list of topics that are discussed in the reviews. Give the topics as a keyword.
            2. Identify the positvie, negative and neutral aspects of the reviews. You will return this as a list of topic keywords and their
                associated values (1 for positive, 0 for neutral, -1 for negative).
            3. Summarize the reviews in a concise manner.
        Return the summary and the identified topics in the following format:
        {{
            "text": "<summary>",
            "topics": [
                {{
                    "keyword": "<topic_keyword>",
                    "value": <1|0|-1>
                }},
                ...
            ]
        }}

        Reviews:
        {reviews_text}
        """
    
    response = model.with_structured_output(Summary).invoke(prompt)
    return {"sample_summaries": [response]}


async def summarize_overall(state: OverallState):
    sample_summaries = state['sample_summaries']

    language = "German"
    
    prompt = f"""
        You will be given a list of analyses of samples of product reviews.
        Generate a summary of the analyses as a short text in {language}. Here is an example for a different product:
        --- Example Start ---
        Kunden sind mit der Qualität, Ergonomie und dem Preis-Leistungs-Verhältnis der Maus zufrieden. Sie beschreiben sie als hochwertig, robust und aus wertigem Kunststoff gefertigt. Die Maus liegt angenehm in der Hand und bietet einen festen Griff, was bei intensiven Spielsituationen ein Vorteil ist. Die Anpassbarkeit, das Gewicht und die Anpassbarkeit mit Gewichtstuning werden ebenfalls gelobt. Einige Kunden haben gemischte Meinungen zur Tastenauswahl, Funktionalität und Größe.
        --- Example End ---
        Each analysis contains a list of keywords and their associated values (1 for positive, 0 for neutral, -1 for negative) that were identified in the reviews.
        You will also generate a new list of keywords in {language} from the analyses and assign them values accordingly.
        Here is the list of analyses:
        {sample_summaries}
    """

    response = model.with_structured_output(Summary).invoke(prompt)
    # Save the response to Meteor
    await save_to_meteor({"art_artid": state["art_artid"], "review_summary": response.dict()})

    return {"overall_summary": response}

def create_graph():
    """Create the state graph for the review summary agent."""
    g = StateGraph(OverallState)
    g.add_node("prepare_data", prepare_data)
    g.add_node("summarize_sample", summarize_sample)
    g.add_node("summarize_overall", summarize_overall)
    g.add_edge(START, "prepare_data")
    g.add_conditional_edges("prepare_data", continue_to_sample_summaries, ["summarize_sample"])
    g.add_edge("summarize_sample", "summarize_overall")
    g.add_edge("summarize_overall", END)
    graph = g.compile()
    graph.name = "Review Summary Agent"
    return graph

graph = create_graph()






