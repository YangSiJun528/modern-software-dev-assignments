import os
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 5

# =============================================================================
# 실험 요약
# =============================================================================
# 목표: mistral-nemo:12b로 "httpstatus" → "sutatsptth" (글자 단위 역순)
#
# 문제: LLM은 글자가 아닌 토큰(subword) 단위로 처리함.
#       "httpstatus" = "http" + "status" 토큰으로 인식 → 글자 역순이 어려움.
#
# 실험 흐름:
#   1. 무관한 단어 20-shot (V1) → 실패
#   2. 글자 분리 "c a t → t a c" (V3) → 실패
#      (토큰→글자 분해를 few-shot으로 학습 불가)
#   3. 연관 단어 + 글자 분리 (V4) → 1/2 성공
#   4. 연관 단어만 (V2) → 가장 높은 성공률 (글자 원본 -> 분리된 글자 토큰 연관성을 LLM이 잘 인식하지 못한듯?)
#   5. 규칙 + adversarial 예시 (V5) → V2와 차이 없음
#
# 결론:
#   - 최선: V2 (타겟과 서브워드를 공유하는 연관 단어 예시)
#   - 모델이 "역순 알고리즘"을 학습하는 게 아니라,
#     예시에서 토큰 조각별 역순 패턴을 직접 조회(패턴 매칭)하는 방식
#
# 참고:
#   - 토크나이저 확인: https://tiktokenizer.vercel.app/
#   - 다른사람이 정답을 소개한 글: https://medium.com/@sami93sami93/how-prompting-techniques-transformed-the-llms-we-use-today-2bf2134c39b0
#   - 1주차 자료 중 프롬프트 설명 자료(우측하단 다국어 옵션에 한국어 있음): https://www.promptingguide.ai/techniques/fewshot
# =============================================================================



# =============================================================================
# V1: 무관한 단어 20-shot
# 결과: 항상 실패
# =============================================================================
SYSTEM_PROMPT_V1 = """
You are a pure character-level transducer. Given a word, output ONLY the reversed word. No explanation.

input: cat output: tac
input: lamp output: pmal
input: house output: esuoh
input: orange output: egnaro
input: printer output: retnirp
input: keyboard output: draobyek
input: elephant output: tnahpele
input: dinosaur output: ruasonid
input: mountain output: niatnuom
input: calendar output: radnelac
input: notebook output: koobeton
input: umbrella output: allerbmu
input: blueberry output: yrrebeulb
input: chocolate output: etalocohc
input: pineapple output: elppaenip
input: adventure output: erutnevda
input: crocodile output: elidocorc
input: boulevard output: draveluob
input: jellyfish output: hsifyllej
input: xylophone output: enohpolyx
"""

# =============================================================================
# V2: 연관 단어 20-shot
# 결과: 높은 확률로 성공, 주로 2번째 시도에서 성공
# =============================================================================
SYSTEM_PROMPT_V2 = """
You are a pure character-level transducer. Given a word, output ONLY the reversed word. No explanation.

input: cat output: tac
input: lamp output: pmal
input: house output: esuoh
input: http output: ptth
input: status output: sutats
input: https output: sptth
input: httpapi output: ipaptth
input: statuses output: sesutats
input: httpserver output: revresptth
input: statuspage output: egapsutats
input: httpbin output: nibptth
input: poststatus output: sutatstsop
input: elephant output: tnahpele
input: keyboard output: draobyek
input: mountain output: niatnuom
input: calendar output: radnelac
input: notebook output: koobeton
input: umbrella output: allerbmu
input: chocolate output: etalocohc
input: adventure output: erutnevda
"""

# =============================================================================
# V3: 글자 분리 + 무관한 단어 20-shot
# 결과: 항상 실패
# =============================================================================
SYSTEM_PROMPT_V3 = """
You are a pure character-level transducer. Given a word, reverse it letter by letter.

Process: split into letters, reverse, join. Output ONLY the final reversed word.

input: cat → c a t → t a c → output: tac
input: lamp → l a m p → p m a l → output: pmal
input: house → h o u s e → e s u o h → output: esuoh
input: orange → o r a n g e → e g n a r o → output: egnaro
input: printer → p r i n t e r → r e t n i r p → output: retnirp
input: keyboard → k e y b o a r d → d r a o b y e k → output: draobyek
input: elephant → e l e p h a n t → t n a h p e l e → output: tnahpele
input: dinosaur → d i n o s a u r → r u a s o n i d → output: ruasonid
input: mountain → m o u n t a i n → n i a t n u o m → output: niatnuom
input: calendar → c a l e n d a r → r a d n e l a c → output: radnelac
input: notebook → n o t e b o o k → k o o b e t o n → output: koobeton
input: umbrella → u m b r e l l a → a l l e r b m u → output: allerbmu
input: blueberry → b l u e b e r r y → y r r e b e u l b → output: yrrebeulb
input: chocolate → c h o c o l a t e → e t a l o c o h c → output: etalocohc
input: pineapple → p i n e a p p l e → e l p p a e n i p → output: elppaenip
input: adventure → a d v e n t u r e → e r u t n e v d a → output: erutnevda
input: crocodile → c r o c o d i l e → e l i d o c o r c → output: elidocorc
input: boulevard → b o u l e v a r d → d r a v e l u o b → output: draveluob
input: jellyfish → j e l l y f i s h → h s i f y l l e j → output: hsifyllej
input: xylophone → x y l o p h o n e → e n o h p o l y x → output: enohpolyx

Output ONLY the reversed word, nothing else."""

# =============================================================================
# V4: 글자 분리 + 연관 단어 20-shot
# 결과: 1/2 빈도로 성공, 3과 5사이 시도에서 성공
# =============================================================================
SYSTEM_PROMPT_V4 = """
You are a pure character-level transducer. Given a word, reverse it letter by letter.

Process: split into letters, reverse, join. Output ONLY the final reversed word.

input: cat → c a t → t a c → output: tac
input: lamp → l a m p → p m a l → output: pmal
input: house → h o u s e → e s u o h → output: esuoh
input: http → h t t p → p t t h → output: ptth
input: status → s t a t u s → s u t a t s → output: sutats
input: https → h t t p s → s p t t h → output: sptth
input: httpapi → h t t p a p i → i p a p t t h → output: ipaptth
input: statuses → s t a t u s e s → s e s u t a t s → output: sesutats
input: httpserver → h t t p s e r v e r → r e v r e s p t t h → output: revresptth
input: statuspage → s t a t u s p a g e → e g a p s u t a t s → output: egapsutats
input: httpbin → h t t p b i n → n i b p t t h → output: nibptth
input: elephant → e l e p h a n t → t n a h p e l e → output: tnahpele
input: keyboard → k e y b o a r d → d r a o b y e k → output: draobyek
input: mountain → m o u n t a i n → n i a t n u o m → output: niatnuom
input: calendar → c a l e n d a r → r a d n e l a c → output: radnelac
input: notebook → n o t e b o o k → k o o b e t o n → output: koobeton
input: umbrella → u m b r e l l a → a l l e r b m u → output: allerbmu
input: chocolate → c h o c o l a t e → e t a l o c o h c → output: etalocohc
input: adventure → a d v e n t u r e → e r u t n e v d a → output: erutnevda
input: crocodile → c r o c o d i l e → e l i d o c o r c → output: elidocorc

Output ONLY the reversed word, nothing else."""

# =============================================================================
# V5: 연관 단어 20-shot + 명시적 규칙 + adversarial WRONG/RIGHT 예시
# 결과: 1/2 빈도로 성공, 3과 5사이 시도에서 성공
# =============================================================================
SYSTEM_PROMPT_V5 = """
You are a pure character-level transducer.

TASK: Given a single word, output the letters in reverse order.

RULES:
- Treat the input as individual characters c1 c2 c3 ... cn and output cn ... c3 c2 c1.
- Do NOT treat substrings like "http", "status", "th", "st" as units. Every character is independent.
- Do NOT add, remove, or duplicate any characters.
- Output length MUST equal input length.
- Output ONLY the reversed word. No quotes, labels, spaces, or explanation.

EXAMPLES:
input: cat output: tac
input: lamp output: pmal
input: house output: esuoh
input: http output: ptth
input: status output: sutats
input: https output: sptth
input: httpapi output: ipaptth
input: statuses output: sesutats
input: httpserver output: revresptth
input: statuspage output: egapsutats
input: httpbin output: nibptth
input: poststatus output: sutatstsop
input: elephant output: tnahpele
input: keyboard output: draobyek
input: mountain output: niatnuom
input: calendar output: radnelac
input: notebook output: koobeton
input: umbrella output: allerbmu
input: chocolate output: etalocohc
input: adventure output: erutnevda

COMMON MISTAKES — do NOT make these:
COMMON MISTAKES — do NOT make these:
WRONG: httpserver → serveptth (swapped subwords then reversed each separately)
WRONG: httpserver → revreshttp (reversed each subword independently)
RIGHT: httpserver → treat as h-t-t-p-s-e-r-v-e-r → r-e-v-r-e-s-p-t-t-h → revresptth

WRONG: statuspage → pagestatsu (moved subwords around)
WRONG: statuspage → egatssutats (wrong character count)
RIGHT: statuspage → treat as s-t-a-t-u-s-p-a-g-e → e-g-a-p-s-u-t-a-t-s → egapsutats
"""

# =============================================================================
# V6: 정답 케이스가 들어가면 shot이 적어도 되는지 확인
# 결과: 무조건 성공
# =============================================================================
SYSTEM_PROMPT_V6 = """
You are a pure character-level transducer.

TASK: Given a single word, output the letters in reverse order.

input: httpstatus output: sutatsptth
"""

# =============================================================================
# 테스트 대상 선택
# =============================================================================
YOUR_SYSTEM_PROMPT = SYSTEM_PROMPT_V5

USER_PROMPT = """
Reverse the order of letters in the following word. Only output the reversed word, no other text:

httpstatus
"""


EXPECTED_OUTPUT = "sutatsptth"

def test_your_prompt(system_prompt: str) -> bool:
    """Run the prompt up to NUM_RUNS_TIMES and return True if any output matches EXPECTED_OUTPUT.

    Prints "SUCCESS" when a match is found.
    """
    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        response = chat(
            model="mistral-nemo:12b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
            options={"temperature": 0.5},
        )
        output_text = response.message.content.strip()
        if output_text.strip() == EXPECTED_OUTPUT.strip():
            print("SUCCESS")
            return True
        else:
            print(f"Expected output: {EXPECTED_OUTPUT}")
            print(f"Actual output: {output_text}")
    return False

if __name__ == "__main__":
    test_your_prompt(YOUR_SYSTEM_PROMPT)
