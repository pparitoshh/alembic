"""Question decoding tests: real Teacher, fake SDK; no inference/network."""
import copy
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from openai.types.chat import ChatCompletion
from pydantic import ValidationError
from distillkit.config import EndpointCfg, GenerateCfg, QuestionRepetitionCfg
from distillkit.schemas import GeneratedQuestion, JudgeVerdict, parse_json, question_transport_schema
from distillkit.teacher import Teacher


class QuestionDecodingTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.responses = []
        self.cfg = EndpointCfg(base_url='http://127.0.0.1:9/v1', model='synthetic', max_tokens=1024,
                               extra_body={'chat_template_kwargs': {'enable_thinking': False}})
        def create(**kwargs):
            self.calls.append(copy.deepcopy(kwargs))
            return self.responses.pop(0)
        self.fake = NS(chat=NS(completions=NS(create=create)))
        self.addCleanup((mock := patch('distillkit.teacher.OpenAI', return_value=self.fake)).stop)
        mock.start()
        self.teacher = Teacher(self.cfg)

    def response(self, text, finish='stop', stop_reason=None):
        r = ChatCompletion.model_validate({'id':'synthetic', 'object':'chat.completion', 'created':0,
            'model':'synthetic', 'choices':[{'index':0, 'finish_reason':'stop',
            'message': {'role':'assistant','content':text}}]})
        # SDK permits vLLM's finish extension; no invented successful termination.
        r.choices[0] = r.choices[0].model_copy(update={'finish_reason':finish, 'stop_reason':stop_reason})
        return r

    def test_request_override_preserves_tokens_model_and_endpoint_defaults(self):
        before = self.cfg.model_dump()
        self.responses = [self.response('{"question":"Which source input matters?"}'), self.response('An answer.')]
        extra = {'repetition_detection': QuestionRepetitionCfg().model_dump()}
        parsed, _ = self.teacher.chat_json('system','user',GeneratedQuestion,extra_body=extra)
        self.assertIsNotNone(parsed)
        self.teacher.complete([{'role':'user','content':'answer request'}])
        self.assertEqual(self.calls[0]['extra_body'], {**before['extra_body'], **extra})
        self.assertEqual(self.calls[1]['extra_body'], before['extra_body'])
        self.assertEqual(self.cfg.model_dump(), before)
        self.assertTrue(all(c['max_tokens']==1024 for c in self.calls))
        self.assertFalse(self.calls[0]['response_format']['json_schema']['schema']['additionalProperties'])
        self.assertNotIn('minLength',self.calls[0]['response_format']['json_schema']['schema']['properties']['question'])

    def test_length_and_repetition_are_never_parsed_or_retried(self):
        for finish, reason in [('length',None), ('repetition','repetition_detected'), ('stop','repetition_detected')]:
            with self.subTest(finish=finish):
                raw = '{"question":"Even complete looking JSON is unusable here."}'
                self.responses=[self.response(raw,finish,reason)]
                count=len(self.calls)
                parsed,captured=self.teacher.chat_json('s','u',GeneratedQuestion)
                self.assertIsNone(parsed);self.assertEqual(captured,raw)
                self.assertEqual(len(self.calls),count+1)

    def test_invalid_complete_json_retains_existing_one_parse_retry(self):
        self.responses=[self.response('bad JSON'),self.response('{"question":"Which source input matters?"}')]
        parsed,_=self.teacher.chat_json('s','u',GeneratedQuestion)
        self.assertIsNotNone(parsed);self.assertEqual(len(self.calls),2)

    def test_question_change_does_not_rebase_the_existing_judge_protocol(self):
        self.responses=[self.response('{"verdict":"A"}',finish='length')]
        parsed,_=self.teacher.chat_json('s','u',JudgeVerdict)
        self.assertEqual(parsed.verdict,'A')  # Existing judge behavior, not an endorsement of it.
        self.assertNotIn('repetition_detection',self.calls[0]['extra_body'])

    def test_only_requested_question_field_is_valid(self):
        self.assertIsNone(parse_json(GeneratedQuestion, '{"question":"Which source input matters?","extra":"unexpected"}'))
        self.assertIsNotNone(parse_json(GeneratedQuestion, '{"question":"Explain the literal `echo hello` example."}'))

    def test_transport_relaxation_does_not_weaken_or_mutate_length_validation(self):
        original=GeneratedQuestion.model_json_schema()
        self.assertEqual(original['properties']['question']['minLength'],10)
        question_transport_schema()
        self.assertEqual(GeneratedQuestion.model_json_schema(),original)
        self.assertIsNone(parse_json(GeneratedQuestion,'{"question":"short"}'))
        text='Explain this code:\n```c\nprintf("hello");\n```'
        self.assertEqual(parse_json(GeneratedQuestion,json.dumps({'question':text})).question,text)

    def test_optin_defaults_and_invalid_bounds(self):
        cfg=GenerateCfg(questions_per_chunk=1,answers_per_question=1,personas=['fixture'],task_types=['concept'])
        self.assertIsNone(cfg.question_repetition_detection)
        for fields in ({'min_count':1},{'max_pattern_size':0},{'max_pattern_size':65},
                       {'min_pattern_size':17,'max_pattern_size':16},{'not_a_vllm_field':1}):
            with self.subTest(fields=fields),self.assertRaises(ValidationError):QuestionRepetitionCfg(**fields)


if __name__=='__main__':unittest.main()
