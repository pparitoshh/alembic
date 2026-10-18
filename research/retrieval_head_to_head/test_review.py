import json
from pathlib import Path
import tempfile
import unittest
from review import load_pairs, summarize


class DirectReviewTests(unittest.TestCase):
    def row(self, verdicts):
        return {'answer_old': 'old', 'answer_new': 'new', 'finish_reason_old': 'stop',
                'finish_reason_new': 'stop', 'judge': {'new_vs_old': verdicts}}

    def test_reverse_position_is_scored_as_new_win(self):
        result = summarize([self.row(['A', 'B'])])
        self.assertEqual(result['new_preference'], 1.)
        self.assertEqual(result['position_consistency'], 1.)

    def test_invalid_does_not_become_tie(self):
        result = summarize([self.row(['A', None])])
        self.assertEqual(result['status'], 'incomplete')
        self.assertIsNone(result['new_preference'])
        self.assertIsNone(result['paired_question_bootstrap_95_percent'])

    def test_question_reference_mismatch_fails(self):
        row = {'id': 'x', 'question': 'q', 'reference': 'r', 'answer_rag': 'answer', 'finish_reason_rag': 'stop'}
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp) / 'old', Path(tmp) / 'new'
            old.write_text(json.dumps(row) + '\n')
            new.write_text(json.dumps({**row, 'reference': 'different'}) + '\n')
            with self.assertRaisesRegex(ValueError, 'reference mismatch'):
                load_pairs(old, new, 1, 'rag')

    def test_all_questions_receive_equal_weight(self):
        result = summarize([self.row(['A', 'B']), self.row(['B', 'A'])])
        self.assertEqual(result['new_preference'], .5)


if __name__ == '__main__':
    unittest.main()
