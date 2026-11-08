import unittest
from unittest.mock import Mock, patch

import chromadb

with patch.object(chromadb, "PersistentClient") as persistent_client:
    persistent_client.return_value.get_collection.return_value = Mock()
    import rag


class RagOutputTests(unittest.TestCase):
    def test_adds_relevant_source_link_to_answer(self):
        hits = [{
            "source": "THWS Faculty FIW",
            "section": "FIW studies > Programme contact",
            "source_url": "https://fiw.thws.de/en/studies/",
        }]

        answer = rag.add_source_link("The advisor is listed on the contacts page.", hits)

        self.assertIn(
            "[FIW studies > Programme contact](https://fiw.thws.de/en/studies/)",
            answer,
        )

    def test_does_not_duplicate_source_link_already_in_answer(self):
        url = "https://fiw.thws.de/en/studies/"
        hits = [{"source": "THWS Faculty FIW", "section": "FIW studies", "source_url": url}]

        answer = rag.add_source_link(f"See {url}", hits)

        self.assertEqual(answer, f"See {url}")


if __name__ == "__main__":
    unittest.main()
