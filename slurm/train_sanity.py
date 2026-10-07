"""Run the production trainer once, capture evidence, save/reload and smoke-test its adapter.

Must run on an allocated GPU. Input is a resolved Config JSON, never a .env file.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'GPU sanity requires a Slurm allocation'
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from peft import PeftModel, get_peft_model_state_dict
    from distillkit.config import Config
    from distillkit.generate import _gold
    from distillkit.io import read_jsonl
    from distillkit.records import training_example
    from distillkit.training_audit import finite_adapter_gradients, inspect_prose_labels, matching_inference_inputs, parameter_hashes
    from distillkit import train

    cfg = Config.model_validate(json.loads(args.config.read_text()))
    assert cfg.train.load_in_4bit and cfg.train.bf16 and not cfg.train.fp16, "sanity reload expects the reviewed BF16 QDoRA configuration"
    for name in ('training_evidence.json','adapter','checkpoints','loader_ready.jsonl'):
        assert not (cfg.run_dir/name).exists(), f'fresh run required: {name}'
    rows = read_jsonl(cfg.run_dir/'verified.jsonl')
    assert len(rows)==8 and all(r.get('mode')=='prose' for r in rows)
    assert not set(cfg.seeds.eval_docs) & {r['doc_id'] for r in rows}
    _gold(cfg,'prose'); _gold(cfg,'tool_trace')
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert (torch.ones(1,device='cuda')+1).item()==2
    evidence={'started_utc':datetime.now(timezone.utc).isoformat(),'job_id':os.environ['SLURM_JOB_ID'],
              'status':'RUNNING','cuda_available':True,'torch':torch.__version__,'cuda_build':torch.version.cuda,
              'device':torch.cuda.get_device_name(),'module_origin':train.__file__,'records':len(rows),
              'dataset_sha256':hashlib.sha256((cfg.run_dir/'verified.jsonl').read_bytes()).hexdigest(),
              'optimizer_steps':[],'loss_logs':[]}
    path=cfg.run_dir/'training_evidence.json'
    assert not path.exists(), 'fresh run required'
    def save():path.write_text(json.dumps(evidence,indent=2)+'\n')
    save(); started=time.monotonic()

    class Audit(TrainerCallback):
        trainer=None
        def before(self, trainer):
            self.trainer=trainer
            from trl.trainer.sft_trainer import _chunked_cross_entropy_loss
            assert trainer.args.loss_type=='chunked_nll', 'revalidate causal shift for a changed loss'
            evidence['causal_loss']={'implementation':'trl.trainer.sft_trainer._chunked_cross_entropy_loss',
                'source_sha256':hashlib.sha256(inspect.getsource(_chunked_cross_entropy_loss).encode()).hexdigest(),
                'shift':'hidden[..., :-1, :] predicts labels[..., 1:]; collator labels remain unshifted',
                'validation':'test_actual_chunked_loss_shifts_unshifted_labels_exactly_once compares loss and gradients with explicit shifted cross entropy'}
            evidence['labels']=inspect_prose_labels(trainer,rows,cfg.student.system_prompt)
            evidence['parameter_counts']={'peft_trainable_and_total':list(trainer.model.get_nb_trainable_parameters()),
                'stored_frozen_elements':sum(p.numel() for p in trainer.model.parameters() if not p.requires_grad)}
            trainable=[n for n,p in trainer.model.named_parameters() if p.requires_grad]
            assert trainable and all('lora_' in n for n in trainable)
            assert not any('lm_head' in n for n in trainable)
            self.adapter_before=parameter_hashes(trainer.model,True)
            self.base_before=parameter_hashes(trainer.model,False)
            evidence['trainable_parameter_names']=trainable
            evidence['base_parameter_digest_before']=hashlib.sha256(json.dumps(self.base_before,sort_keys=True).encode()).hexdigest()
            evidence['effective_training_args']={k:getattr(trainer.args,k) for k in ['num_train_epochs','per_device_train_batch_size','gradient_accumulation_steps','learning_rate','max_length','assistant_only_loss','loss_type','optim']}
            evidence['effective_training_args']['optim']=str(evidence['effective_training_args']['optim'])
            with (cfg.run_dir/'loader_ready.jsonl').open('x') as f:
                for row in rows:f.write(json.dumps(training_example(row,cfg.student.system_prompt))+'\n')
            save()
        def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
            self.grad=finite_adapter_gradients(model)
        def on_optimizer_step(self,args,state,control,model=None,**kwargs):
            skipped=bool(self.trainer.accelerator.optimizer_step_was_skipped)
            evidence['optimizer_steps'].append({'step_event':state.global_step+1,'skipped':skipped,**self.grad})
            save()
        def on_log(self,args,state,control,logs=None,**kwargs):
            if logs and 'loss' in logs:
                assert torch.isfinite(torch.tensor(logs['loss'])), 'nonfinite loss'
                evidence['loss_logs'].append(dict(logs));save()
        def on_train_end(self,args,state,control,model=None,**kwargs):
            after=parameter_hashes(model,True)
            changed=[n for n,h in after.items() if h!=self.adapter_before[n]]
            base_after=parameter_hashes(model,False)
            assert base_after==self.base_before, 'frozen base weights changed'
            assert changed and any(not s['skipped'] for s in evidence['optimizer_steps'])
            evidence.update(global_step=state.global_step,changed_adapter_parameters=changed,
                            successful_optimizer_updates=sum(not s['skipped'] for s in evidence['optimizer_steps']),
                            frozen_base_unchanged=True,base_parameter_digest_after=hashlib.sha256(json.dumps(base_after,sort_keys=True).encode()).hexdigest(),
                            training_peak_allocated_bytes=torch.cuda.max_memory_allocated(),training_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            save()

    audit=Audit()
    try:
        adapter=train.run(cfg,before_train=audit.before,callbacks=[audit])
        model=audit.trainer.model;model.eval();model.gradient_checkpointing_disable()
        tok=AutoTokenizer.from_pretrained(cfg.student.model,local_files_only=True)
        saved_tok=AutoTokenizer.from_pretrained(adapter,local_files_only=True)
        smoke_prompt='Briefly explain why keeping existing PATH entries matters.'
        messages=[{'role':'system','content':cfg.student.system_prompt},{'role':'user','content':smoke_prompt}]
        encoded=matching_inference_inputs(tok,saved_tok,messages).to('cuda')
        evidence['saved_tokenizer_matches_base_inference_template']=True
        with torch.inference_mode():before_logits=model(**encoded).logits[:,-1,:].float().cpu()
        expected={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_quant_storage=torch.bfloat16)
        base=AutoModelForCausalLM.from_pretrained(cfg.student.model,local_files_only=True,dtype=torch.bfloat16,
                                                quantization_config=quant,device_map={'':0})
        reloaded=PeftModel.from_pretrained(base,adapter,is_trainable=False,autocast_adapter_dtype=False)
        reloaded.eval()
        actual=get_peft_model_state_dict(reloaded)
        assert expected.keys()==actual.keys()
        assert all(torch.equal(expected[k],actual[k].detach().cpu()) for k in expected), 'saved adapter tensors differ after reload'
        with torch.inference_mode():
            after_logits=reloaded(**encoded).logits[:,-1,:].float().cpu()
            output=reloaded.generate(**encoded,max_new_tokens=32,do_sample=False,pad_token_id=tok.pad_token_id)
        max_diff=float((before_logits-after_logits).abs().max())
        assert torch.allclose(before_logits,after_logits,atol=0.05,rtol=0.01), f'reload logits differ: {max_diff}'
        text=tok.decode(output[0,encoded['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        assert text, 'empty reload inference'
        evidence.update(status='PASS',adapter_path=str(adapter),adapter_tensors_exact_after_reload=True,
                        reload_logit_max_absolute_difference=max_diff,reload_logit_tolerance={'atol':0.05,'rtol':0.01},
                        inference_smoke={'prompt':smoke_prompt,'completion':text,'max_new_tokens':32},
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as exc:
        evidence.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        evidence.update(elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        save()
    print(json.dumps({k:evidence[k] for k in ['status','global_step','adapter_path','frozen_base_unchanged','adapter_tensors_exact_after_reload','elapsed_seconds']}))


if __name__=='__main__':main()
