import json
from pathlib import Path
import tempfile
import unittest
from review import load_pairs, summarize


class ReviewTests(unittest.TestCase):
    def row(self, verdicts):
        return {'id': 'q1', 'answer_base': 'base', 'answer_rag': 'rag',
                'judge': {'rag_vs_base': verdicts}, 'gptoss_verdicts': ['A', 'B']}

    def test_order_swapped_scoring_and_ties(self):
        result = summarize([self.row(['A', 'B']), self.row(['T', 'T'])])
        self.assertEqual(result['gemma_win_rate'], .75)
        self.assertEqual(result['gemma_position_consistency'], 1)
        self.assertEqual(result['ordered_verdict_agreement'], .5)

    def test_invalid_is_incomplete_not_tie(self):
        result = summarize([self.row(['A', None])])
        self.assertEqual(result['invalid_verdicts'], 1)
        self.assertIsNone(result['gemma_win_rate'])
        self.assertIsNone(result['ordered_verdict_agreement'])

    def test_incomplete_or_duplicate_originals_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'pairs.jsonl'
            p.write_text(json.dumps(self.row(['A', 'B']))+'\n')
            with self.assertRaises(ValueError): load_pairs(p, 2)
            p.write_text((json.dumps(self.row(['A', 'B']))+'\n')*2)
            with self.assertRaises(ValueError): load_pairs(p, 2)


if __name__ == '__main__':
    unittest.main()
