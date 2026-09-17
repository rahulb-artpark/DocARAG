from __future__ import annotations

from typing import Optional
import unittest

from pydantic import BaseModel, ValidationError

from retrieval_engine.providers.structured_output import parse_structured, strip_code_fence


class Entity(BaseModel):
    entity_name: str
    entity_type: Optional[str] = None


class StructuredOutputTests(unittest.TestCase):
    def test_bare_json_is_validated(self):
        result = parse_structured('{"entity_name": "Odoo"}', Entity)

        self.assertEqual("Odoo", result.entity_name)

    def test_fenced_json_is_recovered(self):
        # Ollama does not enforce its schema as a grammar for every model, and
        # an unconstrained one wraps the object in a markdown fence.
        payload = '```json\n{"entity_name": "Odoo", "entity_type": "TECHNOLOGY"}\n```'

        result = parse_structured(payload, Entity)

        self.assertEqual("TECHNOLOGY", result.entity_type)

    def test_yaml_answer_is_recovered(self):
        result = parse_structured("entity_name: Odoo\nentity_type: TECHNOLOGY", Entity)

        self.assertEqual("Odoo", result.entity_name)
        self.assertEqual("TECHNOLOGY", result.entity_type)

    def test_fenced_yaml_answer_is_recovered(self):
        result = parse_structured("```yaml\nentity_name: Odoo\n```", Entity)

        self.assertEqual("Odoo", result.entity_name)

    def test_empty_reply_is_refused(self):
        with self.assertRaisesRegex(ValueError, "empty structured response"):
            parse_structured("   ", Entity)

    def test_prose_is_refused_rather_than_guessed(self):
        with self.assertRaises(ValidationError):
            parse_structured("I could not answer that.", Entity)

    def test_a_schema_violation_still_fails(self):
        with self.assertRaises(ValidationError):
            parse_structured('{"entity_type": "TECHNOLOGY"}', Entity)

    def test_strip_code_fence_leaves_unfenced_text_alone(self):
        self.assertEqual('{"a": 1}', strip_code_fence('{"a": 1}'))
        self.assertEqual('{"a": 1}', strip_code_fence('```json\n{"a": 1}\n```'))


if __name__ == "__main__":
    unittest.main()
