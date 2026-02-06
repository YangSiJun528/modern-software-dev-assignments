import os
import re
from typing import List, Callable
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 5

# =============================================================================
# 실험 요약
# =============================================================================
# 목표: llama3.1:8b에 RAG 기법 적용
#
# 배경: 모델이 학습하지 않은 & 정확한 자료 기반 답변이 필요한 경우,
#       관련 문서를 컨텍스트로 주입하여 정확한 답변을 유도한다.
#
# 실험 흐름:
#   1. YOUR_CONTEXT_PROVIDER에서 corpus를 그대로 반환하도록 변경함 -> 즉시 1회 시도에서 성공함.
#   2. 이전에는 벡터 DB에 문서를 임베딩 저장 후, 유사도 검색으로
#      관련 문서만 추출해 컨텍스트로 제공하는 방식을 사용했었음.
#      핵심은 "관련 데이터를 선별해 컨텍스트로 제공"하는 것이며,
#      선별 방식(전체 전달, 벡터 검색 등)에 관계없이 RAG의 범주에 해당 될 듯
#
# 결론:
#   - 성공. 사실 과제에서 의도한 완전한 해결법은 아닌데, 이미 해봤던거니까 그냥 스킵함
#
# 참고:
#   - 프롬프트 기법: https://www.promptingguide.ai/kr/techniques/rag
#   - LLM 내부 정리: https://gist.github.com/YangSiJun528/5bcb1cf16552a710498bcb82f1dae54e
# =============================================================================

DATA_FILES: List[str] = [
    os.path.join(os.path.dirname(__file__), "data", "api_docs.txt"),
]


def load_corpus_from_files(paths: List[str]) -> List[str]:
    corpus: List[str] = []
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    corpus.append(f.read())
            except Exception as exc:
                corpus.append(f"[load_error] {p}: {exc}")
        else:
            corpus.append(f"[missing_file] {p}")
    return corpus


# Load corpus from external files (simple API docs). If missing, fall back to inline snippet
CORPUS: List[str] = load_corpus_from_files(DATA_FILES)

QUESTION = (
    "Write a Python function `fetch_user_name(user_id: str, api_key: str) -> str` that calls the documented API "
    "to fetch a user by id and returns only the user's name as a string."
)


YOUR_SYSTEM_PROMPT = ""


# For this simple example
# For this coding task, validate by required snippets rather than exact string
REQUIRED_SNIPPETS = [
    "def fetch_user_name(",
    "requests.get",
    "/users/",
    "X-API-Key",
    "return",
]


def YOUR_CONTEXT_PROVIDER(corpus: List[str]) -> List[str]:
    """
    For example, return [] to simulate missing context, or [corpus[0]] to include the API docs.
    """
    return corpus


def make_user_prompt(question: str, context_docs: List[str]) -> str:
    if context_docs:
        context_block = "\n".join(f"- {d}" for d in context_docs)
    else:
        context_block = "(no context provided)"
    return (
        f"Context (use ONLY this information):\n{context_block}\n\n"
        f"Task: {question}\n\n"
        "Requirements:\n"
        "- Use the documented Base URL and endpoint.\n"
        "- Send the documented authentication header.\n"
        "- Raise for non-200 responses.\n"
        "- Return only the user's name string.\n\n"
        "Output: A single fenced Python code block with the function and necessary imports.\n"
    )


def extract_code_block(text: str) -> str:
    """Extract the last fenced Python code block, or any fenced code block, else return text."""
    # Try ```python ... ``` first
    m = re.findall(r"```python\n([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        return m[-1].strip()
    # Fallback to any fenced code block
    m = re.findall(r"```\n([\s\S]*?)```", text)
    if m:
        return m[-1].strip()
    return text.strip()


def test_your_prompt(system_prompt: str, context_provider: Callable[[List[str]], List[str]]) -> bool:
    """Run up to NUM_RUNS_TIMES and return True if any output matches EXPECTED_OUTPUT."""
    context_docs = context_provider(CORPUS)
    user_prompt = make_user_prompt(QUESTION, context_docs)

    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        response = chat(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.0},
        )
        output_text = response.message.content
        code = extract_code_block(output_text)
        missing = [s for s in REQUIRED_SNIPPETS if s not in code]
        if not missing:
            print(output_text)
            print("SUCCESS")
            return True
        else:
            print("Missing required snippets:")
            for s in missing:
                print(f"  - {s}")
            print("Generated code:\n" + code)
    return False


if __name__ == "__main__":
    test_your_prompt(YOUR_SYSTEM_PROMPT, YOUR_CONTEXT_PROVIDER)
