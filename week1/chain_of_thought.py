import os
import re
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 5

# =============================================================================
# 실험 요약
# =============================================================================
# 목표: llama3.1:8b로 여러 단계의 연산이 필요한 수학 문제 풀기
#
# 문제: LLM은 토큰 단위로 생각함. 토큰당 계산량이 고정되었으므로 중간 과정을 펼쳐야 복잡한 연산을 잘 수행할 수 있음.
#
# 실험 흐름:
#   1. AI에게 해당 수학 문제 풀이 부탁하고 그걸 사용 (알았으면 내가 적었을수도 있긴 한데, 오일러 정리를 내가 배웠던가?)
#   2. 예시 2개로 했는데 성공함(시도 횟수는 다양함). 예시 1개는 실패함.
#   3. 사고 과정 말로 설명 + 예시 1개 -> 실패
#   4. 사고 과정 말로 설명 + 예시 2개 -> 실패
#   5. 사고 과정 설명은 딱히 의미가 없네, 시스템 프롬프트는 순수하게 학습자료만 제공해줘야 하나?
#
# 결론:
#   - 충분히 큰 모델에 사고 과정을 전개하면서 풀도록하면 복잡한 수학 문제도 풀 수 있다
#   - 복잡한 multi-step verbal instruction을 따르는 능력은 모델의 성능에 의존함. (검증 필요함. AI 답변인데 2022년도 기준 논문 몇게 기반으로 알려줌)
#     따라서 8b같이 낮은 모델의 경우 in-context learning을 통해 패턴 매칭을 유도하는 것이 더 높은 효율을 보여줄 것임.
#
# 참고:
#   1주차 자료
#   - 프롬프트 설명 자료: https://www.promptingguide.ai/kr/techniques/cot
#   - LLM 내부 정리: https://gist.github.com/YangSiJun528/5bcb1cf16552a710498bcb82f1dae54e
# =============================================================================

# =============================================================================
# V1: COT 예시 1개
# 결과: 항상 실패
# =============================================================================
SYSTEM_PROMPT_V1 = """
what is 7^{9999} (mod 1000)?

Step 1: Euler's theorem

φ(1000) = φ(8)·φ(125) = 4·100 = 400

Since gcd(7, 1000) = 1, we have 7^400 ≡ 1 (mod 1000).

Step 2: Reduce the exponent

9999 = 400 · 24 + 399

So 7^{9999} ≡ 7^{399} (mod 1000).

Step 3: Compute 7^{399} mod 1000 via repeated squaring

Build a power table (mod 1000):

  7^1   = 7
  7^2   = 49
  7^4   = 49^2   = 2401   → 401
  7^8   = 401^2  = 160801 → 801
  7^16  = 801^2  = 641601 → 601
  7^32  = 601^2  = 361201 → 201
  7^64  = 201^2  = 40401  → 401
  7^128 = 401^2  = 160801 → 801
  7^256 = 801^2  = 641601 → 601

Decompose the exponent in binary:
399 = 256 + 128 + 8 + 4 + 2 + 1

Verify: 256 + 128 + 8 + 4 + 2 + 1 = 399 ✓

Step 4: Multiply the corresponding powers

7^{399} = 7^{256} · 7^{128} · 7^{8} · 7^{4} · 7^{2} · 7^{1}
        = 601 · 801 · 801 · 401 · 49 · 7  (mod 1000)

Compute step by step:

  601 · 801 = 481401 → 401
  401 · 801 = 321201 → 201
  201 · 401 = 80601  → 601
  601 · 49  = 29449  → 449
  449 · 7   = 3143   → 143

Answer: 143
"""

# =============================================================================
# V1: COT 예시 2개
# 결과: 항상 실패. 1~5번째 시도 다양하게 성공
# =============================================================================
SYSTEM_PROMPT_V2 = """
Solve the given modular exponentiation problem step by step using the following approach:

1. Apply Euler's theorem (or Fermat's Little Theorem if the modulus is prime) to find the order.
2. Reduce the exponent modulo the order.
3. Compute the remaining power using repeated squaring.
4. Multiply the relevant powers step by step, reducing mod at each multiplication.
5. Give the final answer on the last line as "Answer: <number>".

Here is a worked example:


what is 7^{9999} (mod 1000)?

Step 1: Euler's theorem

φ(1000) = φ(8)·φ(125) = 4·100 = 400

Since gcd(7, 1000) = 1, we have 7^400 ≡ 1 (mod 1000).

Step 2: Reduce the exponent

9999 = 400 · 24 + 399

So 7^{9999} ≡ 7^{399} (mod 1000).

Step 3: Compute 7^{399} mod 1000 via repeated squaring

Build a power table (mod 1000):

  7^1   = 7
  7^2   = 49
  7^4   = 49^2   = 2401   → 401
  7^8   = 401^2  = 160801 → 801
  7^16  = 801^2  = 641601 → 601
  7^32  = 601^2  = 361201 → 201
  7^64  = 201^2  = 40401  → 401
  7^128 = 401^2  = 160801 → 801
  7^256 = 801^2  = 641601 → 601

Decompose the exponent in binary:
399 = 256 + 128 + 8 + 4 + 2 + 1

Verify: 256 + 128 + 8 + 4 + 2 + 1 = 399 ✓

Step 4: Multiply the corresponding powers

7^{399} = 7^{256} · 7^{128} · 7^{8} · 7^{4} · 7^{2} · 7^{1}
        = 601 · 801 · 801 · 401 · 49 · 7  (mod 1000)

Compute step by step:

  601 · 801 = 481401 → 401
  401 · 801 = 321201 → 201
  201 · 401 = 80601  → 601
  601 · 49  = 29449  → 449
  449 · 7   = 3143   → 143

Answer: 143


what is 2^{100000} (mod 97)?

Step 1: Fermat's Little Theorem

97 is prime and gcd(2, 97) = 1, so 2^{96} ≡ 1 (mod 97).

Step 2: Reduce the exponent

100000 = 96 · 1041 + 64

So 2^{100000} ≡ 2^{64} (mod 97).

Step 3: Compute 2^{64} mod 97 via repeated squaring

  2^1  = 2
  2^2  = 4
  2^4  = 16
  2^8  = 256       → 256 - 2·97 = 62
  2^16 = 62^2      = 3844 → 3844 - 39·97 = 3844 - 3783 = 61
  2^32 = 61^2      = 3721 → 3721 - 38·97 = 3721 - 3686 = 35
  2^64 = 35^2      = 1225 → 1225 - 12·97 = 1225 - 1164 = 61

Answer: 61
"""

# =============================================================================
# V1: COT 예시 2개
# 결과: 대부분 성공. 1~5번째 시도 다양하게 성공
# =============================================================================
SYSTEM_PROMPT_V3 = """
what is 7^{9999} (mod 1000)?

Step 1: Euler's theorem

φ(1000) = φ(8)·φ(125) = 4·100 = 400

Since gcd(7, 1000) = 1, we have 7^400 ≡ 1 (mod 1000).

Step 2: Reduce the exponent

9999 = 400 · 24 + 399

So 7^{9999} ≡ 7^{399} (mod 1000).

Step 3: Compute 7^{399} mod 1000 via repeated squaring

Build a power table (mod 1000):

  7^1   = 7
  7^2   = 49
  7^4   = 49^2   = 2401   → 401
  7^8   = 401^2  = 160801 → 801
  7^16  = 801^2  = 641601 → 601
  7^32  = 601^2  = 361201 → 201
  7^64  = 201^2  = 40401  → 401
  7^128 = 401^2  = 160801 → 801
  7^256 = 801^2  = 641601 → 601

Decompose the exponent in binary:
399 = 256 + 128 + 8 + 4 + 2 + 1

Verify: 256 + 128 + 8 + 4 + 2 + 1 = 399 ✓

Step 4: Multiply the corresponding powers

7^{399} = 7^{256} · 7^{128} · 7^{8} · 7^{4} · 7^{2} · 7^{1}
        = 601 · 801 · 801 · 401 · 49 · 7  (mod 1000)

Compute step by step:

  601 · 801 = 481401 → 401
  401 · 801 = 321201 → 201
  201 · 401 = 80601  → 601
  601 · 49  = 29449  → 449
  449 · 7   = 3143   → 143

Answer: 143


what is 2^{100000} (mod 97)?

Step 1: Fermat's Little Theorem

97 is prime and gcd(2, 97) = 1, so 2^{96} ≡ 1 (mod 97).

Step 2: Reduce the exponent

100000 = 96 · 1041 + 64

So 2^{100000} ≡ 2^{64} (mod 97).

Step 3: Compute 2^{64} mod 97 via repeated squaring

  2^1  = 2
  2^2  = 4
  2^4  = 16
  2^8  = 256       → 256 - 2·97 = 62
  2^16 = 62^2      = 3844 → 3844 - 39·97 = 3844 - 3783 = 61
  2^32 = 61^2      = 3721 → 3721 - 38·97 = 3721 - 3686 = 35
  2^64 = 35^2      = 1225 → 1225 - 12·97 = 1225 - 1164 = 61

Answer: 61
"""

YOUR_SYSTEM_PROMPT = SYSTEM_PROMPT_V3


USER_PROMPT = """
Solve this problem, then give the final answer on the last line as "Answer: <number>".

what is 3^{12345} (mod 100)?
"""


# For this simple example, we expect the final numeric answer only
EXPECTED_OUTPUT = "Answer: 43"


def extract_final_answer(text: str) -> str:
    """Extract the final 'Answer: ...' line from a verbose reasoning trace.

    - Finds the LAST line that starts with 'Answer:' (case-insensitive)
    - Normalizes to 'Answer: <number>' when a number is present
    - Falls back to returning the matched content if no number is detected
    """
    matches = re.findall(r"(?mi)^\s*answer\s*:\s*(.+)\s*$", text)
    if matches:
        value = matches[-1].strip()
        # Prefer a numeric normalization when possible (supports integers/decimals)
        num_match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if num_match:
            return f"Answer: {num_match.group(0)}"
        return f"Answer: {value}"
    return text.strip()


def test_your_prompt(system_prompt: str) -> bool:
    """Run up to NUM_RUNS_TIMES and return True if any output matches EXPECTED_OUTPUT.

    Prints "SUCCESS" when a match is found.
    """
    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        response = chat(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
            options={"temperature": 0.3},
        )
        output_text = response.message.content
        final_answer = extract_final_answer(output_text)
        if final_answer.strip() == EXPECTED_OUTPUT.strip():
            print("SUCCESS")
            return True
        else:
            print(f"Expected output: {EXPECTED_OUTPUT}")
            print(f"Actual output: {final_answer}")
    return False


if __name__ == "__main__":
    test_your_prompt(YOUR_SYSTEM_PROMPT)


