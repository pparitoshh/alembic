import unittest
from experiment import input_row, completion_metadata, score_rows


class MatchedComparisonTests(unittest.TestCase):
    def test_reference_and_prior_answers_cannot_enter_inputs(self):
        source = {'id': 'q', 'question': 'Question', 'reference': 'SECRET_REFERENCE',
                  'answer_rag': 'OLD_ANSWER', 'retrieval': [{'url': 'u', 'site': 's', 'retrieved_at': 'd', 'text': 'Source text'}]}
        row = input_row(source)
        self.assertNotIn('SECRET_REFERENCE', str(row))
        self.assertNotIn('OLD_ANSWER', str(row))
        self.assertIn('Source text', row['inference_question'])

    def test_stop_length_and_unknown_are_distinct(self):
        self.assertEqual(completion_metadata([1, 2, 9, 0], {9}, 4)['finish_reason'], 'stop')
        self.assertEqual(completion_metadata([1, 2, 3, 4], {9}, 4)['finish_reason'], 'length')
        self.assertEqual(completion_metadata([1, 2], {9}, 4)['finish_reason'], 'unknown')

    def test_invalid_judgment_is_not_a_tie(self):
        row = {'judge': {'student_rag_vs_base_rag': ['A', None]}}
        self.assertEqual(score_rows([row])['status'], 'incomplete')

    def test_both_orders_use_equal_question_weight(self):
        rows = [{'judge': {'student_rag_vs_base_rag': v}} for v in [['A', 'B'], ['B', 'A'], ['T', 'T']]]
        self.assertEqual(score_rows(rows)['preference'], .5)


if __name__ == '__main__':
    unittest.main()
