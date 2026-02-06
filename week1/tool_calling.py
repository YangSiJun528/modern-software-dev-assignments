import ast
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Callable

from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 3

# =============================================================================
# 실험 요약
# =============================================================================
# 목표: llama3.1:8b로 도구를 사용할 수 있게 프롬프트 작성하기
#
# 문제: LLM은 이러한 커스텀 도구를 사용하는 방법을 모름
#
# 실험 흐름:
#   1. 앞선 활동으로 저성능 LLM은 메타 설명으론 문제 해결이 어렵다고 판단했으므로, 예시를 제공해주는 식으로 방향성을 잡음.
#      >  (저성능 모델은 instruction following보다 pattern matching에 강하다.)
#   2. 코드 이해를 위해 AI로 분석. 도구 설명을 하드코딩 해도 될거 같은데, 추가를 고려해서 TOOL_REGISTRY로 동적으로 생성해야 하나?
#      일단 고정된 문자열로 가능하면 추가하는거야 어렵지 않을거라 판단해서 고정된 문자로 시도.
#   3. 성공. 난이도 자체는 하드코딩 된 문자만 나오게 하면 되서 낮았음.
#
# 결론:
#   - LLM이 외부 서비스를 호출하게 할 수 있다. (Skill.md의 아주 간단한 느낌)
#
# 참고:
#   1주차 자료
#   - LLM 내부 정리: https://gist.github.com/YangSiJun528/5bcb1cf16552a710498bcb82f1dae54e
# =============================================================================


# ==========================
# Tool 구현부 (실제 실행되는 함수들)
# ==========================

def _annotation_to_str(annotation: Optional[ast.AST]) -> str:
    """
    AST 타입 어노테이션 노드를 문자열로 변환하는 헬퍼.
    """
    if annotation is None:
        return "None"
    try:
        # AST 노드를 소스 코드 문자열로 역변환.
        return ast.unparse(annotation)  # type: ignore[attr-defined]
    except Exception:
        # unparse 실패 시 fallback
        if isinstance(annotation, ast.Name):
            return annotation.id
        return type(annotation).__name__


def _list_function_return_types(file_path: str) -> List[Tuple[str, str]]:
    """
    주어진 Python 파일을 파싱하여 최상위(top-level) 함수들의
    (함수명, 리턴타입 문자열) 리스트를 반환.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)  # 소스코드 → AST(추상 구문 트리) 변환
    results: List[Tuple[str, str]] = []
    for node in tree.body:
        # 최상위 레벨의 함수 정의만 수집
        if isinstance(node, ast.FunctionDef):
            return_str = _annotation_to_str(node.returns)  # 리턴 타입 어노테이션
            results.append((node.name, return_str))
    results.sort(key=lambda x: x[0])  # 함수명 기준 정렬
    return results


def output_every_func_return_type(file_path: str = None) -> str:
    """
    [등록된 Tool]
    Python 파일의 모든 최상위 함수에 대해 "함수명: 리턴타입" 형태의
    줄바꿈 구분 문자열을 반환한다.
    """
    path = file_path or __file__
    if not os.path.isabs(path):
        # 상대경로인 경우, 이 스크립트 기준으로 해석 시도
        candidate = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(candidate):
            path = candidate
    pairs = _list_function_return_types(path)
    return "\n".join(f"{name}: {ret}" for name, ret in pairs)


# ----------------------------------------------------------
# 샘플 함수들: 파일 내에 분석 대상이 존재하도록 하기 위한 더미 함수
# ----------------------------------------------------------
def add(a: int, b: int) -> int:
    """두 정수를 더해서 반환."""
    return a + b


def greet(name: str) -> str:
    """인사 문자열을 반환."""
    return f"Hello, {name}!"


# ----------------------------------------------------------
# Tool 레지스트리: 문자열 이름 → 함수 매핑
# LLM이 출력한 tool 이름으로 실제 함수를 찾아 실행하는 데 사용
# ----------------------------------------------------------
TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "output_every_func_return_type": output_every_func_return_type,
}


# ==========================
# 프롬프트 설정
# ==========================

SYSTEM_PROMPT_V1 = """
You ONLY output a JSON object. No explanation, no markdown, no code fences, no extra text.

Available tool:
- output_every_func_return_type(file_path): Returns every function's return type in a Python file. Empty string uses the default file.

When the user says "Call the tool now.", respond with EXACTLY this JSON and nothing else:

{"tool": "output_every_func_return_type", "args": {"file_path": ""}}

Example 1:
User: Call the tool now.
Assistant: {"tool": "output_every_func_return_type", "args": {"file_path": ""}}

Example 2:
User: Call the tool now.
Assistant: {"tool": "output_every_func_return_type", "args": {"file_path": "/home/user/main.py"}}
"""

YOUR_SYSTEM_PROMPT = SYSTEM_PROMPT_V1

# ==========================
# 유틸리티 함수들
# ==========================

def resolve_path(p: str) -> str:
    """
    경로 해석 헬퍼.
    절대경로면 그대로 반환, 상대경로면 이 스크립트 디렉토리 기준으로 해석.
    """
    if os.path.isabs(p):
        return p
    here = os.path.dirname(__file__)
    c1 = os.path.join(here, p)
    if os.path.exists(c1):
        return c1
    return p


def extract_tool_call(text: str) -> Dict[str, Any]:
    """
    모델 출력 텍스트에서 JSON 객체를 파싱.

    모델이 ```json ... ``` 코드펜스로 감싸는 경우도 처리.
    파싱 실패 시 ValueError 발생.

    기대하는 JSON 형식:
        {"tool": "함수이름", "args": {"key": "value", ...}}
    """
    text = text.strip()
    # 코드펜스 제거
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json\n"):
            text = text[5:]
    try:
        obj = json.loads(text)
        return obj
    except json.JSONDecodeError:
        raise ValueError("Model did not return valid JSON for the tool call")


# ==========================
# 모델 호출 및 실행
# ==========================

def run_model_for_tool_call(system_prompt: str) -> Dict[str, Any]:
    """
    Ollama를 통해 llama3.1:8b 모델을 호출하고,
    응답에서 tool call JSON을 파싱하여 반환.

    메시지 구성:
        system: 사용자가 작성한 시스템 프롬프트
        user:   "Call the tool now."  (고정)

    temperature=0.3 으로 비교적 결정적인(deterministic) 출력 유도.
    """
    response = chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Call the tool now."},
        ],
        options={"temperature": 0.3},
    )
    content = response.message.content
    return extract_tool_call(content)


def execute_tool_call(call: Dict[str, Any]) -> str:
    """
    파싱된 tool call 딕셔너리를 받아 실제 함수를 실행하고 결과를 반환.

    매개변수 call의 기대 형식:
        {
            "tool": "output_every_func_return_type",
            "args": {"file_path": "경로"}  # args는 선택적
        }

    처리 로직:
        1. "tool" 키로 TOOL_REGISTRY에서 함수를 찾음
        2. "args"가 있으면 kwargs로 전달
        3. file_path가 빈 문자열이면 __file__로 대체
        4. file_path가 아예 없으면 __file__을 기본값으로 주입
    """
    name = call.get("tool")
    if not isinstance(name, str):
        raise ValueError("Tool call JSON missing 'tool' string")
    func = TOOL_REGISTRY.get(name)
    if func is None:
        raise ValueError(f"Unknown tool: {name}")
    args = call.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("Tool call JSON 'args' must be an object")

    # file_path 인자 처리: 경로 해석 또는 기본값 설정
    if "file_path" in args and isinstance(args["file_path"], str):
        args["file_path"] = resolve_path(args["file_path"]) if str(args["file_path"]) != "" else __file__
    elif "file_path" not in args:
        args["file_path"] = __file__  # 기본값: 이 스크립트 자신

    return func(**args)


def compute_expected_output() -> str:
    """
    Ground truth(정답) 생성.
    이 파일 자체를 대상으로 output_every_func_return_type를 직접 실행하여
    기대 출력값을 계산한다.
    """
    return output_every_func_return_type(__file__)


# ==========================
# 메인 테스트 함수
# ==========================

def test_your_prompt(system_prompt: str) -> bool:
    """
    시스템 프롬프트 검증 테스트.

    절차:
        1. compute_expected_output()로 정답을 미리 계산
        2. 최대 NUM_RUNS_TIMES(3)번 반복:
           a. LLM에 프롬프트를 보내 tool call JSON을 받음
           b. 해당 JSON으로 실제 tool을 실행
           c. 실행 결과와 정답을 비교
           d. 일치하면 SUCCESS 출력 후 True 반환
        3. 3번 모두 실패하면 False 반환

    성공 조건:
        - LLM이 유효한 JSON을 출력해야 함
        - JSON의 "tool" 값이 TOOL_REGISTRY에 존재해야 함
        - tool 실행 결과가 expected와 정확히 일치해야 함
    """
    expected = compute_expected_output()
    for _ in range(NUM_RUNS_TIMES):
        # 1단계: LLM 호출 → JSON 파싱
        try:
            call = run_model_for_tool_call(system_prompt)
        except Exception as exc:
            print(f"Failed to parse tool call: {exc}")
            continue  # 파싱 실패 시 재시도

        print(call)

        # 2단계: tool 실행
        try:
            actual = execute_tool_call(call)
        except Exception as exc:
            print(f"Tool execution failed: {exc}")
            continue  # 실행 실패 시 재시도

        # 3단계: 결과 비교
        if actual.strip() == expected.strip():
            print(f"Generated tool call: {call}")
            print(f"Generated output: {actual}")
            print("SUCCESS")
            return True
        else:
            print("Expected output:\n" + expected)
            print("Actual output:\n" + actual)

    return False  # 3번 모두 실패


if __name__ == "__main__":
    test_your_prompt(YOUR_SYSTEM_PROMPT)