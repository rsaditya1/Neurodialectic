from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
import os
import re
import json
import logging
from datetime import datetime

# ---------------- SETUP ----------------

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

groq_key = os.getenv("GROQ_SCAR_KEY")
cohere_key = os.getenv("COHERE_API_KEY")

if not all([groq_key, cohere_key]):
    raise ValueError("Missing API keys")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LIMITS ----------------

MAX_PROMPT_CHARS = 6000
MAX_MEMORY_CHARS = 1000
MAX_SUMMARY_CHARS = 1800

# ---------------- MODELS ----------------

generator_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_key,
    temperature=0.6
)

critic_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_key,
    temperature=0.3
)

validator_llm = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=groq_key,
    temperature=0.1
)

summarizer_llm = ChatCohere(
    api_key=cohere_key,
    temperature=0.3
)

# ---------------- MEMORY ----------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="neurodialectic_memory",
    embedding_function=embedding_model,
    persist_directory="./memory_db"
)

# ---------------- CONFIDENCE PARSER ----------------

def parse_confidence(text: str) -> float:
    """
    Robustly extract a confidence score from messy LLM output.
    Handles:
      - <think>...</think> wrapper tags (qwen3)
      - "Confidence: 0.85"
      - "Confidence: 85%"
      - "confidence_score: 0.9"
      - "0.72" on its own line
      - Markdown bold like **0.85**
    Returns float between 0.0 and 1.0, defaults to 0.5 on failure.
    """

    # Strip <think>...</think> blocks entirely
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = cleaned.strip()

    if not cleaned:
        # Everything was inside <think> tags — try parsing from original
        cleaned = text

    # Strategy 1: Look for explicit "Confidence: X" pattern
    patterns = [
        r"[Cc]onfidence[\s_]*(?:[Ss]core)?[\s:=]*\*?\*?([0-9]+(?:\.[0-9]+)?)\s*%",
        r"[Cc]onfidence[\s_]*(?:[Ss]core)?[\s:=]*\*?\*?([0-9]*\.?[0-9]+)\s*(?:/\s*1(?:\.0)?)?\*?\*?",
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            value = float(match.group(1))
            if value > 1.0:
                value = value / 100.0
            return max(0.0, min(1.0, value))

    # Strategy 2: Find any decimal between 0 and 1 in the cleaned text
    decimals = re.findall(r"\b(0\.\d+|1\.0)\b", cleaned)
    if decimals:
        return float(decimals[0])

    # Strategy 3: Find any percentage
    percentages = re.findall(r"(\d{1,3})\s*%", cleaned)
    if percentages:
        value = float(percentages[0]) / 100.0
        return max(0.0, min(1.0, value))

    logger.warning(f"Could not parse confidence from validator output. First 200 chars: {cleaned[:200]}")
    return 0.5


# ---------------- WORKFLOW ----------------

def create_workflow(query, max_iterations=5):

    class GraphState(TypedDict):
        query: str
        draft: Optional[str]
        critique: Optional[str]
        validation: Optional[str]
        confidence: float
        iteration: int
        final_answer: Optional[str]
        summary: Optional[str]
        generator_output: Optional[str]
        critic_output: Optional[str]
        validator_output: Optional[str]
        refinement_outputs: List[str]

    workflow = StateGraph(GraphState)

    # -------- GENERATOR --------
    def generator_node(state):
        try:
            memories = vector_store.similarity_search(state["query"], k=1)

            memory_context = ""
            for doc in memories:
                memory_context += doc.page_content[:500] + "\n"
            memory_context = memory_context.strip()[:MAX_MEMORY_CHARS]

            if memory_context:
                prompt = f"""Answer clearly with structured reasoning.

Question:
{state['query']}

Avoid repeating these past failure patterns:
{memory_context}"""
            else:
                prompt = f"""Answer clearly with structured reasoning.

Question:
{state['query']}"""

            draft = generator_llm.invoke(prompt[:MAX_PROMPT_CHARS]).content
            logger.info(f"Generator produced {len(draft)} chars")

            return {
                "draft": draft,
                "generator_output": draft,
                "iteration": 0,
                "refinement_outputs": []
            }

        except Exception as e:
            logger.error(f"Generator failed: {e}")
            fallback = f"I encountered an error generating a response: {str(e)}"
            return {
                "draft": fallback,
                "generator_output": fallback,
                "iteration": 0,
                "refinement_outputs": []
            }

    # -------- CRITIC --------
    def critic_node(state):
        try:
            prompt = f"""You are a critical reviewer. Analyze the following answer for:
1. Factual accuracy
2. Logical consistency
3. Completeness
4. Clarity

Be specific about what is wrong and what could be improved.

Answer to critique:
{state['draft']}"""

            critique = critic_llm.invoke(prompt[:MAX_PROMPT_CHARS]).content
            logger.info(f"Critic produced {len(critique)} chars")

            return {
                "critique": critique,
                "critic_output": critique
            }

        except Exception as e:
            logger.error(f"Critic failed: {e}")
            fallback = f"Critique unavailable due to error: {str(e)}"
            return {
                "critique": fallback,
                "critic_output": fallback
            }

    # -------- VALIDATOR --------
    def validator_node(state):
        try:
            prompt = f"""You are a strict answer quality evaluator.

Rate the answer below on a scale from 0.0 to 1.0 based on accuracy, completeness, and clarity.

ANSWER:
{state['draft'][:2000]}

CRITIQUE:
{state['critique'][:2000]}

You MUST respond in EXACTLY this format and nothing else:

Confidence: <number between 0.0 and 1.0>
Reason: <one sentence explanation>

Do NOT include any other text, thinking, or preamble. Just those two lines."""

            response = validator_llm.invoke(prompt[:MAX_PROMPT_CHARS]).content

            logger.info(f"Raw validator response ({len(response)} chars): {response[:300]}")

            confidence = parse_confidence(response)

            # Clean the response for display — strip <think> tags
            display_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            if not display_response:
                display_response = f"Confidence: {confidence}\nReason: Extracted from model reasoning"

            logger.info(f"Parsed confidence: {confidence}")

            return {
                "validation": display_response,
                "confidence": confidence,
                "validator_output": display_response
            }

        except Exception as e:
            logger.error(f"Validator failed: {e}")
            return {
                "validation": f"Validation error: {str(e)}",
                "confidence": 0.5,
                "validator_output": f"Validation error: {str(e)}"
            }

    # -------- CONTROLLER --------
    def controller(state):
        confidence = state.get("confidence", 0)
        iteration = state.get("iteration", 0)

        logger.info(f"Controller check — confidence: {confidence}, iteration: {iteration}/{max_iterations}")

        if confidence >= 0.85:
            logger.info("Confidence threshold met, finalizing")
            return "finalize"
        if iteration >= max_iterations:
            logger.info("Max iterations reached, finalizing")
            return "finalize"

        logger.info("Sending to refinement")
        return "refine"

    # -------- REFINE --------
    def refine_node(state):
        try:
            prompt = f"""Improve the answer below based on the critique provided.
Keep what is already correct. Fix what is wrong. Fill in what is missing.

Original Answer:
{state['draft']}

Critique:
{state['critique']}

Provide the improved answer:"""

            improved = generator_llm.invoke(prompt[:MAX_PROMPT_CHARS]).content

            # Create a new list to avoid state mutation issues
            previous = list(state.get("refinement_outputs", []))
            previous.append(improved)

            current_iteration = state.get("iteration", 0) + 1
            logger.info(f"Refinement {current_iteration} produced {len(improved)} chars")

            return {
                "draft": improved,
                "iteration": current_iteration,
                "refinement_outputs": previous
            }

        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return {
                "iteration": state.get("iteration", 0) + 1
            }

    # -------- FINALIZE --------
    def finalize_node(state):
        try:
            if state.get("confidence", 1.0) < 0.85:
                compressed = summarizer_llm.invoke(
                    f"Summarize this critique as a failure pattern in under 120 words:\n{state.get('critique', 'No critique')}"
                ).content[:500]

                vector_store.add_documents([
                    Document(page_content=f"Failure Pattern: {compressed}")
                ])

                logger.info("Stored failure pattern in memory")

        except Exception as e:
            logger.warning(f"Failed to store memory (non-fatal): {e}")

        return {"final_answer": state.get("draft", "No answer generated")}

    # -------- SUMMARIZER --------
    def summarizer_node(state):
        try:
            prompt = f"""Summarize the full reasoning process concisely.

Final Answer:
{state.get('final_answer', 'N/A')[:1500]}

Critique:
{state.get('critique', 'N/A')[:1000]}

Validation:
{state.get('validation', 'N/A')[:500]}

Include:
- Conclusion
- Strengths
- Weaknesses
- Final confidence: {state.get('confidence', 'N/A')}"""

            summary = summarizer_llm.invoke(prompt).content
            summary = summary[:MAX_SUMMARY_CHARS]

            return {"summary": summary}

        except Exception as e:
            logger.error(f"Summarizer failed: {e}")
            return {"summary": f"Summary generation failed: {str(e)}"}

    # -------- GRAPH --------
    workflow.add_node("generator", generator_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("refine", refine_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("summarizer", summarizer_node)

    workflow.set_entry_point("generator")
    workflow.add_edge("generator", "critic")
    workflow.add_edge("critic", "validator")

    workflow.add_conditional_edges(
        "validator",
        controller,
        {"refine": "refine", "finalize": "finalize"}
    )

    workflow.add_edge("refine", "critic")
    workflow.add_edge("finalize", "summarizer")
    workflow.add_edge("summarizer", END)

    return workflow

# ---------------- TERMINAL EXECUTION ----------------

if __name__ == "__main__":

    print("\n===== NeuraDialectic Terminal Mode =====\n")

    query = input("Enter your question:\n> ")
    max_iterations_input = input("Max refinement iterations (default 5): ")

    try:
        max_iterations = int(max_iterations_input)
    except:
        max_iterations = 5

    workflow = create_workflow(query, max_iterations)
    result = workflow.compile().invoke({"query": query})

    print("\n===== FINAL ANSWER =====\n")
    print(result.get("final_answer"))

    print("\n===== SUMMARY =====\n")
    print(result.get("summary"))

    print("\n===== GENERATOR OUTPUT =====\n")
    print(result.get("generator_output"))

    print("\n===== CRITIC OUTPUT =====\n")
    print(result.get("critic_output"))

    print("\n===== VALIDATOR OUTPUT =====\n")
    print(result.get("validator_output"))

    print("\n===== REFINEMENT STEPS =====\n")
    for i, r in enumerate(result.get("refinement_outputs", []), 1):
        print(f"\n--- Refinement {i} ---\n")
        print(r)

    print("\n===== CONFIDENCE =====\n")
    print(result.get("confidence"))

    # Save run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(OUTPUT_DIR, f"neurodialectic_{timestamp}.json")

    run_data = {
        "metadata": {
            "timestamp": timestamp,
            "query": query
        },
        "outputs": result
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=4, ensure_ascii=False)

    print(f"\nRun saved to {file_path}")