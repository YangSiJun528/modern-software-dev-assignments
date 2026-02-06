import os
import re
from typing import Callable, List, Tuple
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 1

# =============================================================================
# 실험 요약
# =============================================================================
# 목표: llama3.1:8b에 Reflexion 기법 적용
#
# 배경: 실패를 언어적 피드백으로 받아 재시도하면 문제 해결 능력이 올라감
#
# 실험 흐름:
#   1. Reflexion이 뭐지 -> Reflection의 엣날 단어, 의미는 동일, AI 쪽에서 이 기법을 설명할 때 씀.
#   2. 실행 흐름이 어떻게 되지. run_reflexion_flow만 보면 됨. LLM이 출력한 코드를 파이썬이 실행함.
#   3. 재실행 프롬프트랑 동적인 내용 채우는 your_build_reflexion_context를 완성하면 됨.
#   4. 그냥 적절하게 추가하니까 해결됨.
#
# 결론:
#   - 딱히 어려울건 없었음.
#   - 실제로 에이전트가 함수나 기능 구현하고 테스트하고 버그 있으면 고치는 것도 동일한 식으로 (개별 호출/컨텍스트에 추가되서) 동작할 듯
#
# 참고:
#   - 프롬프트 기법: https://www.promptingguide.ai/kr/techniques/reflexion
#   - LLM 내부 정리: https://gist.github.com/YangSiJun528/5bcb1cf16552a710498bcb82f1dae54e
# =============================================================================


SYSTEM_PROMPT = """
You are a coding assistant. Output ONLY a single fenced Python code block that defines
the function is_valid_password(password: str) -> bool. No prose or comments.
Keep the implementation minimal.
"""

YOUR_REFLEXION_PROMPT = """
You are a coding assistant. Given the previous code and test failures,
output ONLY a corrected Python code block defining is_valid_password(password: str) -> bool.
No prose or comments.
"""

# Ground-truth test suite used to evaluate generated code
SPECIALS = set("!@#$%^&*()-_")
TEST_CASES: List[Tuple[str, bool]] = [
    ("Password1!", True),       # valid
    ("password1!", False),      # missing uppercase
    ("Password!", False),       # missing digit
    ("Password1", False),       # missing special
]


def extract_code_block(text: str) -> str:
    m = re.findall(r"```python\n([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        return m[-1].strip()
    m = re.findall(r"```\n([\s\S]*?)```", text)
    if m:
        return m[-1].strip()
    return text.strip()


def load_function_from_code(code_str: str) -> Callable[[str], bool]:
    namespace: dict = {}
    exec(code_str, namespace)  # noqa: S102 (executing controlled code from model for exercise)
    func = namespace.get("is_valid_password")
    if not callable(func):
        raise ValueError("No callable is_valid_password found in generated code")
    return func


def evaluate_function(func: Callable[[str], bool]) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    for pw, expected in TEST_CASES:
        try:
            result = bool(func(pw))
        except Exception as exc:
            failures.append(f"Input: {pw} → raised exception: {exc}")
            continue

        if result != expected:
            # Compute diagnostic based on ground-truth rules
            reasons = []
            if len(pw) < 8:
                reasons.append("length < 8")
            if not any(c.islower() for c in pw):
                reasons.append("missing lowercase")
            if not any(c.isupper() for c in pw):
                reasons.append("missing uppercase")
            if not any(c.isdigit() for c in pw):
                reasons.append("missing digit")
            if not any(c in SPECIALS for c in pw):
                reasons.append("missing special")
            if any(c.isspace() for c in pw):
                reasons.append("has whitespace")

            failures.append(
                f"Input: {pw} → expected {expected}, got {result}. Failing checks: {', '.join(reasons) or 'unknown'}"
            )

    return (len(failures) == 0, failures)


def generate_initial_function(system_prompt: str) -> str:
    response = chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Provide the implementation now."},
        ],
        options={"temperature": 0.2},
    )
    return extract_code_block(response.message.content)


def your_build_reflexion_context(prev_code: str, failures: List[str]) -> str:
    return (
        f"Previous code:\n```python\n{prev_code}\n```\n\n"
        f"Test failures:\n" + "\n".join(f"- {f}" for f in failures)
    )


def apply_reflexion(
    reflexion_prompt: str,
    build_context: Callable[[str, List[str]], str],
    prev_code: str,
    failures: List[str],
) -> str:
    reflection_context = build_context(prev_code, failures)
    print(f"REFLECTION CONTEXT: {reflection_context}, {reflexion_prompt}")
    response = chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": reflexion_prompt},
            {"role": "user", "content": reflection_context},
        ],
        options={"temperature": 0.2},
    )
    return extract_code_block(response.message.content)


def run_reflexion_flow(
    system_prompt: str,
    reflexion_prompt: str,
    build_context: Callable[[str, List[str]], str],
) -> bool:
    # 1) 초기 함수 생성 — LLM에게 system_prompt를 주고 is_valid_password 코드를 생성시킨다
    initial_code = generate_initial_function(system_prompt)
    print("Initial code:\n" + initial_code)

    # 생성된 코드 문자열을 exec으로 실행하여 실제 호출 가능한 함수 객체로 변환
    func = load_function_from_code(initial_code)

    # 4개의 테스트 케이스로 함수를 평가. passed=전체 통과 여부, failures=실패 목록
    passed, failures = evaluate_function(func)

    if passed:
        # 초기 생성 코드가 모든 테스트를 통과하면 바로 성공 반환 (테스트하면서 이런 적은 없었는데, 드문 경우 있을수도?)
        print("SUCCESS (initial implementation passed all tests)")
        return True
    else:
        # 실패한 테스트가 있으면 실패 내역 출력 후 Reflexion 단계로 진행
        print(f"FAILURE (initial implementation failed some tests): {failures}")

    # 2) Reflexion 1회 수행 — 이전 코드와 실패 정보를 LLM에 넘겨 수정된 코드를 받는다
    improved_code = apply_reflexion(reflexion_prompt, build_context, initial_code, failures)
    print("\nImproved code:\n" + improved_code)

    # 수정된 코드를 다시 함수 객체로 변환
    improved_func = load_function_from_code(improved_code)

    # 수정된 함수를 동일한 테스트 케이스로 재평가
    passed2, failures2 = evaluate_function(improved_func)

    if passed2:
        # Reflexion 후 모든 테스트 통과 시 성공
        print("SUCCESS")
        return True

    # Reflexion 후에도 여전히 실패하는 테스트 출력 후 실패 반환
    print("Tests still failing after reflexion:")
    for f in failures2:
        print("- " + f)
    return False


if __name__ == "__main__":
    run_reflexion_flow(SYSTEM_PROMPT, YOUR_REFLEXION_PROMPT, your_build_reflexion_context)
