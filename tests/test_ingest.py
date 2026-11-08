import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ingest


class CsvIngestionTests(unittest.TestCase):
    def test_build_chunks_reads_csv_records_and_source_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            csv_path = data_dir / "knowledge.csv"
            with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "record_id", "question", "answer", "source_id",
                        "section", "source_url", "source_last_verified",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "record_id": "KB-001",
                    "question": "How do I contact support?",
                    "answer": "Email the support team.",
                    "source_id": "THWS ITSC",
                    "section": "Student services",
                    "source_url": "https://example.edu/support",
                    "source_last_verified": "2026-11-07",
                })

            with patch.object(ingest.config, "DATA_DIR", data_dir):
                chunks, total, records = ingest.build_chunks()

        self.assertEqual(total, 1)
        self.assertEqual(records, 1)
        chunk = next(iter(chunks.values()))
        self.assertIn("Question: How do I contact support?", chunk["text"])
        self.assertIn("Answer: Email the support team.", chunk["text"])
        self.assertEqual(chunk["source"], "THWS ITSC")
        self.assertEqual(chunk["record_id"], "KB-001")
        self.assertEqual(chunk["section"], "Student services")
        self.assertEqual(chunk["source_url"], "https://example.edu/support")
        self.assertEqual(chunk["source_last_verified"], "2026-11-07")


if __name__ == "__main__":
    unittest.main()
