from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
trl = pytest.importorskip('trl')

from distillkit import train
from distillkit.config import Config
from distillkit.io import write_jsonl
from distillkit.training_audit import finite_adapter_gradients, inspect_prose_labels, parameter_hashes
from test_pipeline import _cfg_dict


class CharacterTokenizer:
    """Synthetic role-boundary fixture only; real tokenizer checks run in the GPU driver."""
    eos_token = '~'
    eos_token_id = ord('~')
    pad_token_id = 0
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False,**kw):
        text=''.join(f"{m['role']}:{m['content']}~\n" for m in messages)
        return text+('assistant:' if add_generation_prompt else '')
    def __call__(self,text,**kw):
        return {'input_ids':list(map(ord,text)),'offset_mapping':[(i,i+1) for i in range(len(text))]}
    def decode(self,ids):return ''.join(map(chr,ids))
    def convert_ids_to_tokens(self,t):return chr(t)


def label_fixture():
    tok=CharacterTokenizer();rows=[];prepared=[]
    for i,answer in enumerate(['A.','Long answer.']):
        rows.append({'id':str(i),'messages':[{'role':'user','content':'Q?'},{'role':'assistant','content':answer}]})
        prefix='system:SYS~\nuser:Q?~\nassistant:'
        ids=list(map(ord,prefix+answer+'~\n'))
        prepared.append({'input_ids':ids,'labels':[-100]*len(prefix)+ids[len(prefix):]})
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    trainer=SimpleNamespace(processing_class=tok,chat_template='synthetic',train_dataset=prepared,
                            args=SimpleNamespace(max_length=128,loss_type='chunked_nll'),
                            data_collator=DataCollatorForLanguageModeling(pad_token_id=0))
    return trainer,rows


def test_mask_check_includes_eos_and_template_newline_and_masks_padding():
    trainer,rows=label_fixture();r=inspect_prose_labels(trainer,rows,'SYS')
    assert r['all_padding_masked'] and r['padding_tokens']>0
    assert r['rows'][0]['supervised_tokens']==4  # A. + EOS + newline


@pytest.mark.parametrize('defect',['user_loss','dropped_eos','shifted_labels','truncation'])
def test_mask_check_rejects_real_training_data_defects(defect):
    trainer,rows=label_fixture();p=trainer.train_dataset[0]
    if defect=='user_loss':p['labels'][0]=p['input_ids'][0]
    elif defect=='dropped_eos':p['labels'][-2]=-100
    elif defect=='shifted_labels':p['labels']=p['labels'][1:]+[-100]
    else:p['input_ids']=p['input_ids'][:-2];p['labels']=p['labels'][:-2]
    with pytest.raises(AssertionError):inspect_prose_labels(trainer,rows,'SYS')


def test_parameter_and_gradient_checks_detect_change_and_frozen_leak():
    model=torch.nn.Linear(2,2);model.bias.requires_grad=False
    before=parameter_hashes(model,True)
    model.weight.grad=torch.ones_like(model.weight)
    assert finite_adapter_gradients(model)['nonzero']
    with torch.no_grad():model.weight.add_(0.1)
    assert parameter_hashes(model,True)!=before
    model.bias.grad=torch.ones_like(model.bias)
    with pytest.raises(AssertionError,match='frozen'):finite_adapter_gradients(model)


def test_actual_chunked_loss_shifts_unshifted_labels_exactly_once():
    from trl.trainer.sft_trainer import _chunked_cross_entropy_loss
    torch.manual_seed(17)
    hidden=torch.randn(1,5,3,requires_grad=True)
    head=torch.randn(7,3,requires_grad=True)
    labels=torch.tensor([[-100,-100,2,4,1]])
    actual=_chunked_cross_entropy_loss(hidden_states=hidden,lm_head_weight=head,chunk_size=2,labels=labels)[0]
    logits=hidden@head.T
    expected=torch.nn.functional.cross_entropy(logits[:,:-1,:].reshape(-1,7),labels[:,1:].reshape(-1),ignore_index=-100)
    assert torch.allclose(actual,expected,atol=1e-6)
    a=torch.autograd.grad(actual,(hidden,head),retain_graph=True)
    b=torch.autograd.grad(expected,(hidden,head))
    assert all(torch.allclose(x,y,atol=1e-6) for x,y in zip(a,b))


@pytest.mark.parametrize('abort',[False,True])
def test_production_run_audit_precedes_optimizer_and_can_stop_it(tmp_path,monkeypatch,abort):
    import transformers
    d=_cfg_dict(tmp_path);d['train'].update(load_in_4bit=False,bf16=False,fp16=False)
    cfg=Config.model_validate(d);cfg.run_dir.mkdir()
    write_jsonl(cfg.run_dir/'verified.jsonl',[{'question':'Synthetic Q','answer':'Synthetic A'}])
    events=[]
    class Tokenizer:
        def save_pretrained(self,*a):events.append('tokenizer_saved')
        def apply_chat_template(self,messages,**k):return ' '.join(m['content'] for m in messages)
        def __call__(self,text,**k):return {'input_ids':text.split()}
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:Tokenizer())
    class Trainer:
        def __init__(self,**kw):
            assert kw['args'].assistant_only_loss
            assert kw['train_dataset'][0]['messages'][0]['role']=='system'
            events.append('constructed')
        def add_callback(self,c):events.append('callback')
        def train(self,resume_from_checkpoint=None):events.append('optimized')
        def save_model(self,p):events.append('saved')
    monkeypatch.setattr(trl,'SFTTrainer',Trainer)
    def inspect(t):
        events.append('inspected')
        if abort:raise RuntimeError('bad labels')
    if abort:
        with pytest.raises(RuntimeError,match='bad labels'):train.run(cfg,before_train=inspect,callbacks=[object()])
        assert events==['constructed','callback','inspected']
    else:
        assert train.run(cfg,before_train=inspect,callbacks=[object()])==cfg.run_dir/'adapter'
        assert events==['constructed','callback','inspected','optimized','saved','tokenizer_saved']


def test_inference_comparison_handles_batchencoding_default_and_detects_changes():
    from transformers import BatchEncoding
    from distillkit.training_audit import matching_inference_inputs
    class Tokenizer:
        chat_template='same';eos_token_id=2;pad_token_id=0
        ids=[1,3,2]
        def apply_chat_template(self,messages,**kwargs):
            assert kwargs['return_dict'] is True and kwargs['return_tensors']=='pt'
            return BatchEncoding({'input_ids':torch.tensor([self.ids]),'attention_mask':torch.ones(1,len(self.ids),dtype=torch.long)})
    base=Tokenizer();saved=Tokenizer()
    assert matching_inference_inputs(base,saved,[])['input_ids'].tolist()==[[1,3,2]]
    saved.ids=[1,4,2]
    with pytest.raises(AssertionError,match='prompt IDs'):matching_inference_inputs(base,saved,[])
