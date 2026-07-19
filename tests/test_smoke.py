import os
import sys
import unittest

# 确保能导入 src 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.schemas import ClarifierOutput, ReviewFeedback, JudgeOutput
from src.models.llm_factory import _safe_json_parse, _normalize_structured_data
from src.agents.graph import build_graph


class TestSchemas(unittest.TestCase):
    def test_clarifier_output(self):
        out = ClarifierOutput(is_clear=True, response="重构后的问题", reasoning="已足够清晰")
        self.assertTrue(out.is_clear)
        self.assertEqual(out.response, "重构后的问题")

    def test_review_feedback_pass(self):
        fb = ReviewFeedback(score=9, comments="逻辑清晰", passed=True)
        self.assertTrue(fb.passed)
        self.assertEqual(fb.score, 9)

    def test_review_feedback_fail(self):
        fb = ReviewFeedback(score=6, comments="需要补充", passed=False)
        self.assertFalse(fb.passed)


class TestJsonParsing(unittest.TestCase):
    def test_parse_standard_clarifier(self):
        text = '{"is_clear": false, "response": "请问 1. ... 2. ..."}'
        parsed = _safe_json_parse(text, ClarifierOutput)
        self.assertIsNotNone(parsed)
        self.assertFalse(parsed.is_clear)
        self.assertIn("请问", parsed.response)

    def test_parse_fenced_json(self):
        text = '```json\n{"is_clear": true, "response": "ok"}\n```'
        parsed = _safe_json_parse(text, ClarifierOutput)
        self.assertIsNotNone(parsed)
        self.assertTrue(parsed.is_clear)

    def test_parse_alternative_keys(self):
        # 模型未按 schema 输出时的容错
        data = {
            "status": "needs_clarification",
            "clarifying_questions": ["q1", "q2"],
        }
        normalized = _normalize_structured_data(data, ClarifierOutput)
        parsed = ClarifierOutput.model_validate(normalized)
        self.assertFalse(parsed.is_clear)
        self.assertIn("q1", parsed.response)


class TestGraphCompilation(unittest.TestCase):
    def test_build_graph(self):
        app = build_graph()
        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main()
