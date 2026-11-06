"""The agent loop: the model picks a tool, we run it, the result goes back, repeat."""
import json
import re

from llm_client import GeminiClient, LLMError

from .tools import TOOLS

MAX_STEPS = 6              # hard cap so the agent can never loop forever
MAX_OBSERVATION = 3000     # characters of tool output sent back to the model
CONSOLE_PREVIEW = 400      # characters of tool output printed to the console


class AgentError(Exception):
    """Raised when the agent cannot continue (for example the API keeps failing)."""


def _tool_guide() -> str:
    return "\n".join(f"- {name}: {t['description']}" for name, t in TOOLS.items())


SYSTEM_INSTRUCTION = (
    "You are an agent that solves tasks step by step using tools.\n\n"
    "Available tools:\n{tools}\n\n"
    "At every step reply with ONE JSON object and nothing else.\n"
    'To use a tool: {"thought": "why", "action": "<tool name>", "input": "<tool input>"}\n'
    'To finish: {"thought": "why", "final_answer": "<your answer>"}\n\n'
    "Rules:\n"
    "- Use one tool per step, then wait for the Observation.\n"
    "- Never invent facts or numbers. Get them from a tool.\n"
    "- Use the calculator for any arithmetic. Do not calculate in your head.\n"
    "- If a tool returns an Error, try a different input or a different tool.\n"
    "- If you cannot do exactly what was asked (for example a file does not exist), say so "
    "clearly in the final_answer. Never silently substitute something else.\n"
    "- Use the earlier conversation to understand follow-up questions.\n"
    "- If the user says their information is in an uploaded document (such as a resume), "
    "call list_files, then read_file on the right file, before saying you do not know.\n"
    "- When you have enough information, give the final_answer."
).replace("{tools}", _tool_guide())


def _parse(raw: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def _call_model(client: GeminiClient, prompt: str) -> str:
    try:
        return client.generate(prompt, system_instruction=SYSTEM_INSTRUCTION, json_mode=True)
    except LLMError as e:
        raise AgentError(f"Model call failed: {e}") from e


def run_agent(question: str, client: GeminiClient | None = None,
              max_steps: int = MAX_STEPS, verbose: bool = True, on_step=None,
              history: list | None = None) -> dict:
    """Run the agent. Returns {"answer": str, "steps": [...], "finished": bool}.

    on_step is an optional function called with each step as soon as it finishes,
    so a user interface can show progress live.
    history is an optional list of earlier chat messages ({"role", "content"}),
    so the agent can understand follow-up questions.
    """
    if not question or not question.strip():
        raise ValueError("Question is empty.")

    client = client or GeminiClient()
    transcript = ""
    if history:
        lines = []
        for m in history[-6:]:                      # only the last few messages
            who = "User" if m.get("role") == "user" else "Assistant"
            lines.append(f"{who}: {str(m.get('content', ''))[:600]}")
        transcript += "Earlier conversation:\n" + "\n".join(lines) + "\n\n"
    transcript += f"Question: {question.strip()}\n"
    steps = []

    for step in range(1, max_steps + 1):
        raw = _call_model(client, transcript + "\nWhat is your next step? Reply with one JSON object.")

        try:
            data = _parse(raw)
        except (json.JSONDecodeError, ValueError) as e:
            if verbose:
                print(f"\n[Step {step}] The model's reply was not valid JSON. Asking again.")
            transcript += f"\nSystem: your last reply was not valid JSON ({e}). Reply with one JSON object.\n"
            continue

        thought = str(data.get("thought", "")).strip()
        if verbose:
            print(f"\n[Step {step}] Thought: {thought}")

        if "final_answer" in data:
            answer = str(data["final_answer"]).strip()
            if verbose:
                print(f"\nFinal answer: {answer}")
            return {"answer": answer, "steps": steps, "finished": True}

        action = data.get("action")
        tool_input = str(data.get("input", ""))
        if action not in TOOLS:
            observation = f"Error: unknown tool '{action}'. Available tools: {', '.join(TOOLS)}"
        else:
            try:
                observation = TOOLS[action]["function"](tool_input)
            except Exception as e:  # tools should not raise, but never let one crash the agent
                observation = f"Error: tool '{action}' failed: {e}"

        if len(observation) > MAX_OBSERVATION:
            observation = observation[:MAX_OBSERVATION] + "\n...[truncated]"

        if verbose:
            print(f"[Step {step}] Action: {action}({tool_input!r})")
            preview = observation if len(observation) <= CONSOLE_PREVIEW else observation[:CONSOLE_PREVIEW] + "..."
            print(f"[Step {step}] Observation: {preview}")

        steps.append({"step": step, "thought": thought, "action": action,
                      "input": tool_input, "observation": observation})
        if on_step:
            on_step(steps[-1])
        transcript += (f"\nThought: {thought}\nAction: {action}\n"
                       f"Input: {tool_input}\nObservation: {observation}\n")

    # Out of steps: ask once for a best-effort answer from what was gathered.
    if verbose:
        print(f"\nReached the limit of {max_steps} steps. Asking for a best-effort answer.")
    raw = _call_model(
        client,
        transcript + "\nYou are out of steps. Give your best final answer now using only the "
                     'observations above. Reply with JSON: {"final_answer": "..."}',
    )
    try:
        answer = str(_parse(raw).get("final_answer", "")).strip()
    except (json.JSONDecodeError, ValueError):
        answer = ""
    answer = answer or "I could not finish the task within the step limit."
    if verbose:
        print(f"\nFinal answer (step limit reached): {answer}")
    return {"answer": answer, "steps": steps, "finished": False}