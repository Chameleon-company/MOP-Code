"""
Evaluation test set for the Smart Street Lighting system.

Contains known-answer questions derived from AS/NZS 1158, energy benchmarks,
and Melbourne data. Used to evaluate:
1. RAG retrieval quality (does the right chunk appear?)
2. LLM answer correctness (does the answer match known facts?)
3. Calculation engine accuracy (do numbers match expected values?)
4. Prompt strategy comparison (zero-shot vs few-shot vs CoT)
"""

# Each test case has:
#   - query: the natural language question
#   - expected_facts: list of facts that must appear in a correct answer
#   - category: which knowledge area it tests
#   - difficulty: easy/medium/hard

TEST_SET = [
    # === AS/NZS 1158 Standards Knowledge ===
    {
        "id": "STD-01",
        "query": "What is the required average illuminance for a P3 category pedestrian area?",
        "expected_facts": ["7 lux", "average"],
        "expected_value": 7.0,
        "category": "standards",
        "difficulty": "easy",
    },
    {
        "id": "STD-02",
        "query": "What P-category should be used for a busy park pathway with high evening pedestrian activity?",
        "expected_facts": ["P2", "P3"],
        "category": "standards",
        "difficulty": "medium",
    },
    {
        "id": "STD-03",
        "query": "What is the minimum illuminance required for a P9 park path?",
        "expected_facts": ["1.0 lux", "minimum"],
        "expected_value": 1.0,
        "category": "standards",
        "difficulty": "easy",
    },
    {
        "id": "STD-04",
        "query": "What uniformity ratio is required for P-category pedestrian lighting?",
        "expected_facts": ["0.5", "Emin/Eavg"],
        "category": "standards",
        "difficulty": "medium",
    },
    {
        "id": "STD-05",
        "query": "What vertical illuminance is required at pedestrian crossings according to AS/NZS 1158.4?",
        "expected_facts": ["50 lux", "1.5"],
        "category": "standards",
        "difficulty": "hard",
    },
    {
        "id": "STD-06",
        "query": "What is the recommended colour temperature for street lighting in Melbourne?",
        "expected_facts": ["3000K", "warm white"],
        "category": "standards",
        "difficulty": "easy",
    },
    {
        "id": "STD-07",
        "query": "Can lighting be dimmed during low-activity periods under AS/NZS 1158?",
        "expected_facts": ["yes", "reduced", "lower P-category"],
        "category": "standards",
        "difficulty": "medium",
    },

    # === Energy & Cost Knowledge ===
    {
        "id": "NRG-01",
        "query": "What is the typical energy saving when replacing HPS street lights with LEDs?",
        "expected_facts": ["50", "70", "percent"],
        "category": "energy",
        "difficulty": "easy",
    },
    {
        "id": "NRG-02",
        "query": "What is Victoria's carbon emission factor for electricity?",
        "expected_facts": ["1.08", "0.96", "CO2"],
        "category": "energy",
        "difficulty": "medium",
    },
    {
        "id": "NRG-03",
        "query": "Is standalone solar street lighting viable for Melbourne?",
        "expected_facts": ["marginal", "winter", "1.6"],
        "category": "energy",
        "difficulty": "medium",
    },
    {
        "id": "NRG-04",
        "query": "How many hours per year do street lights operate in Melbourne?",
        "expected_facts": ["4200", "4,200"],
        "expected_value": 4200,
        "category": "energy",
        "difficulty": "easy",
    },
    {
        "id": "NRG-05",
        "query": "What is the typical payback period for an LED street light upgrade?",
        "expected_facts": ["4", "7", "year"],
        "category": "energy",
        "difficulty": "medium",
    },

    # === Melbourne Context ===
    {
        "id": "MEL-01",
        "query": "What ecological lighting guidelines does Melbourne follow for parks?",
        "expected_facts": ["3000K", "warm", "wildlife", "ecological"],
        "category": "melbourne",
        "difficulty": "medium",
    },
    {
        "id": "MEL-02",
        "query": "What is Melbourne's renewable energy target for council operations?",
        "expected_facts": ["net-zero", "2040", "renewable"],
        "category": "melbourne",
        "difficulty": "hard",
    },

    # === Calculation Verification ===
    {
        "id": "CALC-01",
        "query": "How many LED lights are needed for a 200m park pathway with moderate use?",
        "expected_facts": ["13", "P9", "16"],
        "expected_value": 13,
        "category": "calculation",
        "difficulty": "easy",
    },
    {
        "id": "CALC-02",
        "query": "What is the annual energy cost for 13 x 30W LED lights operating 4200 hours at $0.20/kWh?",
        "expected_facts": ["327", "328"],
        "expected_value": 327.60,
        "category": "calculation",
        "difficulty": "medium",
    },
    {
        "id": "CALC-03",
        "query": "Design lighting for a 300m high-traffic shared pathway. How many lights and what spacing?",
        "expected_facts": ["P2", "P3"],
        "category": "calculation",
        "difficulty": "hard",
    },
]


def get_test_set(category=None, difficulty=None):
    """Filter test set by category and/or difficulty."""
    tests = TEST_SET
    if category:
        tests = [t for t in tests if t["category"] == category]
    if difficulty:
        tests = [t for t in tests if t["difficulty"] == difficulty]
    return tests


def evaluate_answer(answer: str, test_case: dict) -> dict:
    """
    Evaluate whether an answer contains the expected facts.

    Returns:
        dict with: matched_facts, missing_facts, score (0-1), pass (bool)
    """
    answer_lower = answer.lower()
    expected = test_case["expected_facts"]

    matched = []
    missing = []
    for fact in expected:
        if fact.lower() in answer_lower:
            matched.append(fact)
        else:
            missing.append(fact)

    # Score: at least half the facts must be present to pass
    score = len(matched) / len(expected) if expected else 0
    passed = score >= 0.5

    return {
        "test_id": test_case["id"],
        "query": test_case["query"],
        "matched_facts": matched,
        "missing_facts": missing,
        "score": round(score, 2),
        "pass": passed,
    }
