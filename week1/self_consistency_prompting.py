import os
import re
from collections import Counter
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 5

# =============================================================================
# 실험 요약
# =============================================================================
# 목표: llama3.1:8b로 Self-Consistency 기법 사용하기
#
# 문제: temperature가 높은 경우, 오답 확률이 높아짐.
#
# 실험 흐름:
#   1. 그냥 USER_PROMPT에 맞는 답변을 유도하고, 정답 확률을 높이기만 하면 됨.
#   2. COT + few shot으로 하면 어렵지 않게 가능할 듯. 참고 자료 예시가 잘 되어있어서 그거 가져다가 AI써서 약간 가공
#   3. Q, A, Answer로 구성했더니 문장형태로 답변을 해버림 -> Answer를 지우고 A의 마무리 문장을 `Answer: <number>` 형식으로 대체
#   4. 문장 형태 답변이 종종 나오지만, 대부분 1번 정도만 나옴. 항상 2개 이상 정답이 나와서 성공 (2~4개가 정답)
#   5. 가능하면 100% 되게 해보라 해서 개선 시도
#   6. 맨 처음에 정답만 답변하라고 함 -> 효과 없거나 적음. 오히려 정답율 미묘하게 낮아지는거 같음.
#   7. 일단 해결을 했으니 AI 도움 받기 -> fewshot은 충분하다는 피드백
#   8. 문구를 개선해보기로 함. 잘 됨. 5번 성공하는 경우도 있음.
#   9. few shot 늘려보는 것도 한 번 시도 -> 5개를 10개로 -> 별 차이 없음. (2~4개가 정답)
#   10. 근데 버전 나눠서며 실행해보니까 temperature가 1이라 그런가 성공하긴 하는데, 4/5 성공을 보장하는 건 어려운 듯. 일단 성공했으니까... 여기까지
#
# 결론:
#   - Temperature가 높은 경우 환각이나 잘못된 답변을 막기 위해서 Self-Consistency 기법을 쓰는게 효과적이다.
#   - 오히려 프롬프트가 100%로 보장하는 경우가 없었다는게, Self-Consistency 기법의 필요성을 증명하는게 아닌가 싶다.
#
# 참고:
#   1주차 자료
#   - 프롬프트 설명 자료: https://www.promptingguide.ai/kr/techniques/consistency
#   - LLM 내부 정리: https://gist.github.com/YangSiJun528/5bcb1cf16552a710498bcb82f1dae54e
# =============================================================================

SYSTEM_PROMPT_V1 = """
Q: There are 18 apples in a basket. Sam took some apples out of the basket. Now there are 7 apples left in the basket. How many apples did Sam take out?
A: There were 18 apples in the basket to start. Now there are only 7 apples left. The difference is the number Sam took out. So Sam took out 18 - 7 = 11 apples. Answer: 11

Q: A train had 45 passengers. At the first station, 12 passengers got off. At the second station, 8 more passengers got on. How many passengers are on the train now?
A: The train started with 45 passengers. At the first station, 12 got off, so there were 45 - 12 = 33 passengers. Then 8 more got on at the second station. So now there are 33 + 8 = 41 passengers. Answer: 41

Q: Emily had 14 stickers and her friend gave her 9 more. She then gave 6 stickers to her brother. How many stickers does Emily have now?
A: Emily started with 14 stickers. Her friend gave her 9 more, so she had 14 + 9 = 23 stickers. She then gave 6 to her brother. So now she has 23 - 6 = 17 stickers. Answer: 17

Q: A bookshelf has 3 shelves. The top shelf holds 8 books, the middle shelf holds 11 books, and the bottom shelf holds 6 books. How many books are on the bookshelf in total?
A: The top shelf has 8 books, the middle shelf has 11 books, and the bottom shelf has 6 books. So the total number of books is 8 + 11 + 6 = 25 books. Answer: 25

Q: Jake walked a total of 50 blocks to school. He stopped at the library after walking 18 blocks. He stopped again at the park, which was 14 blocks before the school. How many blocks did he walk between the library and the park?
A: Jake's total trip is 50 blocks. He stopped at the library after 18 blocks. The park is 14 blocks before the school, so the park is at 50 - 14 = 36 blocks from the start. The distance between the library and the park is 36 - 18 = 18 blocks. Answer: 18
"""

SYSTEM_PROMPT_V2 = """
Give the final answer on the last line in exactly this format: Answer: <number>


Q: There are 18 apples in a basket. Sam took some apples out of the basket. Now there are 7 apples left in the basket. How many apples did Sam take out?
A: There were 18 apples in the basket to start. Now there are only 7 apples left. The difference is the number Sam took out. So Sam took out 18 - 7 = 11 apples. Answer: 11

Q: A train had 45 passengers. At the first station, 12 passengers got off. At the second station, 8 more passengers got on. How many passengers are on the train now?
A: The train started with 45 passengers. At the first station, 12 got off, so there were 45 - 12 = 33 passengers. Then 8 more got on at the second station. So now there are 33 + 8 = 41 passengers. Answer: 41

Q: Emily had 14 stickers and her friend gave her 9 more. She then gave 6 stickers to her brother. How many stickers does Emily have now?
A: Emily started with 14 stickers. Her friend gave her 9 more, so she had 14 + 9 = 23 stickers. She then gave 6 to her brother. So now she has 23 - 6 = 17 stickers. Answer: 17

Q: A bookshelf has 3 shelves. The top shelf holds 8 books, the middle shelf holds 11 books, and the bottom shelf holds 6 books. How many books are on the bookshelf in total?
A: The top shelf has 8 books, the middle shelf has 11 books, and the bottom shelf has 6 books. So the total number of books is 8 + 11 + 6 = 25 books. Answer: 25

Q: Jake walked a total of 50 blocks to school. He stopped at the library after walking 18 blocks. He stopped again at the park, which was 14 blocks before the school. How many blocks did he walk between the library and the park?
A: Jake's total trip is 50 blocks. He stopped at the library after 18 blocks. The park is 14 blocks before the school, so the park is at 50 - 14 = 36 blocks from the start. The distance between the library and the park is 36 - 18 = 18 blocks. Answer: 18
"""

SYSTEM_PROMPT_V3 = """
Q: There are 18 apples in a basket. Sam took some apples out of the basket. Now there are 7 apples left in the basket. How many apples did Sam take out?
A: There were 18 apples in the basket to start. Now there are only 7 apples left. The difference is the number Sam took out. So Sam took out 18 - 7 = 11 apples. Answer: 11

Q: A train had 45 passengers. At the first station, 12 passengers got off. At the second station, 8 more passengers got on. How many passengers are on the train now?
A: The train started with 45 passengers. At the first station, 12 got off, so there were 45 - 12 = 33 passengers. Then 8 more got on at the second station. So now there are 33 + 8 = 41 passengers. Answer: 41

Q: Emily had 14 stickers and her friend gave her 9 more. She then gave 6 stickers to her brother. How many stickers does Emily have now?
A: Emily started with 14 stickers. Her friend gave her 9 more, so she had 14 + 9 = 23 stickers. She then gave 6 to her brother. So now she has 23 - 6 = 17 stickers. Answer: 17

Q: A bookshelf has 3 shelves. The top shelf holds 8 books, the middle shelf holds 11 books, and the bottom shelf holds 6 books. How many books are on the bookshelf in total?
A: The top shelf has 8 books, the middle shelf has 11 books, and the bottom shelf has 6 books. So the total number of books is 8 + 11 + 6 = 25 books. Answer: 25

Q: Jake walked a total of 50 blocks to school. He stopped at the library after walking 18 blocks. He stopped again at the park, which was 14 blocks before the school. How many blocks did he walk between the library and the park?
A: Jake's total trip is 50 blocks. He stopped at the library after 18 blocks. The park is 14 blocks before the school, so the park is at 50 - 14 = 36 blocks from the start. The distance between the library and the park is 36 - 18 = 18 blocks. Answer: 18

Q: A farmer had 30 chickens. He sold 9 chickens in the morning and bought 5 more in the afternoon. How many chickens does the farmer have now?
A: The farmer started with 30 chickens. He sold 9, so he had 30 - 9 = 21 chickens. Then he bought 5 more, so now he has 21 + 5 = 26 chickens. Answer: 26

Q: Lisa had 40 marbles. She gave 13 marbles to Tom and 8 marbles to Anna. How many marbles does Lisa have left?
A: Lisa started with 40 marbles. She gave away 13 + 8 = 21 marbles in total. So she has 40 - 21 = 19 marbles left. Answer: 19

Q: A parking lot has 3 rows. The first row has 7 cars, the second row has 12 cars, and the third row has 5 cars. 6 cars then leave the parking lot. How many cars are left?
A: The total number of cars is 7 + 12 + 5 = 24 cars. Then 6 cars leave. So there are 24 - 6 = 18 cars left. Answer: 18

Q: Ben ran 8 miles on Monday and 6 miles on Tuesday. His goal for the week was 25 miles. How many more miles does Ben need to run to reach his goal?
A: Ben ran 8 + 6 = 14 miles so far. His goal is 25 miles. So he needs 25 - 14 = 11 more miles. Answer: 11

Q: A store had 55 toys. In the morning, 18 toys were sold. In the afternoon, 10 more toys were delivered. How many toys does the store have now?
A: The store started with 55 toys. 18 were sold, so there were 55 - 18 = 37 toys. Then 10 were delivered, so now there are 37 + 10 = 47 toys. Answer: 47
"""

YOUR_SYSTEM_PROMPT = SYSTEM_PROMPT_V2

USER_PROMPT = """
Solve this problem, then give the final answer on the last line as "Answer: <number>".

Henry made two stops during his 60-mile bike trip. He first stopped after 20
miles. His second stop was 15 miles before the end of the trip. How many miles
did he travel between his first and second stops?
"""

EXPECTED_OUTPUT = "Answer: 25"


def extract_final_answer(text: str) -> str:
    """Extract the final 'Answer: ...' line from a verbose reasoning trace.

    - Finds the LAST line that starts with 'Answer:' (case-insensitive)
    - Normalizes to 'Answer: <number>' when a number is present
    - Falls back to returning the matched content if no number is detected
    """
    matches = re.findall(r"(?mi)^\s*answer\s*:\s*(.+)\s*$", text)
    if matches:
        value = matches[-1].strip()
        num_match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if num_match:
            return f"Answer: {num_match.group(0)}"
        return f"Answer: {value}"
    return text.strip()


def test_your_prompt(system_prompt: str) -> bool:
    """Run the prompt NUM_RUNS_TIMES, majority-vote on the extracted 'Answer: ...' lines.

    Prints "SUCCESS" if the majority answer equals EXPECTED_OUTPUT.
    """
    answers: list[str] = []
    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        response = chat(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
            options={"temperature": 1},
        )
        output_text = response.message.content
        final_answer = extract_final_answer(output_text)
        print(f"Run {idx + 1} answer: {final_answer}")
        answers.append(final_answer.strip())

    if not answers:
        print("No answers produced.")
        return False

    counts = Counter(answers)
    majority_answer, majority_count = counts.most_common(1)[0]
    print(f"Majority answer: {majority_answer} ({majority_count}/{len(answers)})")

    if majority_answer.strip() == EXPECTED_OUTPUT.strip():
        print("SUCCESS")
        return True

    # Print distribution for debugging when majority does not match expected
    print(f"Expected output: {EXPECTED_OUTPUT}")
    print("Answer distribution:")
    for answer, count in counts.most_common():
        print(f"  {answer}: {count}")
    return False


if __name__ == "__main__":
    test_your_prompt(YOUR_SYSTEM_PROMPT)


