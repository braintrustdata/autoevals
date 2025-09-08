import unittest
from dataclasses import dataclass
from typing import List, Optional

from .serializable_data_class import SerializableDataClass


@dataclass
class PromptData(SerializableDataClass):
    prompt: Optional[str] = None
    options: Optional[dict] = None


@dataclass
class PromptSchema(SerializableDataClass):
    id: str
    project_id: str
    _xact_id: str
    name: str
    slug: str
    description: Optional[str]
    prompt_data: PromptData
    tags: Optional[List[str]]


class TestSerializableDataClass(unittest.TestCase):
    def test_from_dict_deep_with_none_values(self):
        """Test that from_dict_deep correctly handles None values in nested objects."""
        test_dict = {
            "id": "456",
            "project_id": "123",
            "_xact_id": "789",
            "name": "test-prompt",
            "slug": "test-prompt",
            "description": None,
            "prompt_data": {"prompt": None, "options": None},
            "tags": None,
        }

        prompt = PromptSchema.from_dict_deep(test_dict)

        # Verify all fields were set correctly.
        self.assertEqual(prompt.id, "456")
        self.assertEqual(prompt.project_id, "123")
        self.assertEqual(prompt._xact_id, "789")
        self.assertEqual(prompt.name, "test-prompt")
        self.assertEqual(prompt.slug, "test-prompt")
        self.assertIsNone(prompt.description)
        self.assertIsNone(prompt.tags)

        # Verify nested object was created and its fields are None.
        self.assertIsInstance(prompt.prompt_data, PromptData)
        self.assertIsNone(prompt.prompt_data.prompt)
        self.assertIsNone(prompt.prompt_data.options)

        # Verify round-trip serialization works.
        round_trip = PromptSchema.from_dict_deep(prompt.as_dict())
        self.assertEqual(round_trip.as_dict(), test_dict)


if __name__ == "__main__":
    unittest.main()
