"""Opt-in evidence checks for a bounded real-model prose training sanity run."""
import hashlib

from .records import training_example


def parameter_hashes(model, trainable):
    """Hash one tensor at a time; do not retain a second model in host memory."""
    import torch
    result = {}
    for name, param in model.named_parameters():
        if param.requires_grad != trainable:
            continue
        tensor = param.detach().contiguous().cpu()
        h = hashlib.sha256(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}".encode())
        h.update(tensor.view(torch.uint8).numpy().tobytes())
        result[name] = h.hexdigest()
    return result


def inspect_prose_labels(trainer, rows, system_prompt):
    """Check actual prepared labels/collator against role boundaries and full tokenization."""
    tok, template = trainer.processing_class, trainer.chat_template
    assert len(trainer.train_dataset) == len(rows), "preprocessing dropped records"
    evidence = []
    for row, prepared in zip(rows, trainer.train_dataset, strict=True):
        ex = training_example(row, system_prompt)
        assert [m['role'] for m in ex['messages']] == ['system', 'user', 'assistant']
        kwargs = dict(chat_template=template, **ex['chat_template_kwargs'])
        full = tok.apply_chat_template(ex['messages'], tokenize=False, **kwargs)
        prefix = tok.apply_chat_template(ex['messages'][:-1], tokenize=False, add_generation_prompt=True, **kwargs)
        assert full.startswith(prefix)
        encoded = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded['input_ids'], encoded['offset_mapping']
        assert prepared['input_ids'] == ids, "truncation or unexpected token transformation"
        assert len(ids) <= trainer.args.max_length
        labels = prepared['labels']
        assert len(labels) == len(ids)
        answer_end = len(prefix) + len(ex['messages'][-1]['content'])
        assert full[len(prefix):answer_end] == ex['messages'][-1]['content']
        assert full[answer_end:] == tok.eos_token + '\n', "unexpected assistant turn suffix"
        expected_targets = []
        for i, ((start, end), token, label) in enumerate(zip(offsets, ids, labels, strict=True)):
            assert not (start < len(prefix) < end), "token straddles user/assistant boundary"
            # This model's training template supervises the answer, im_end AND its newline.
            expected = token if start >= len(prefix) else -100
            assert label == expected, f"{row['id']}: token {i} has wrong role/loss ownership"
            if label != -100:
                expected_targets.append(i)
        terminal_positions = [i for i in expected_targets if ids[i] == tok.eos_token_id]
        assert expected_targets and len(terminal_positions) == 1
        terminal_position = terminal_positions[0]
        assert tok.decode([ids[i] for i in expected_targets if i > terminal_position]).strip() == ''
        assert all(labels[i] == -100 for i, (_, end) in enumerate(offsets) if end <= len(prefix))
        # Labels remain aligned with input ids here. The selected causal loss shifts once.
        assert all(labels[i] == ids[i] for i in expected_targets)
        evidence.append({'id':row['id'],'tokens':len(ids),'supervised_tokens':len(expected_targets),
                         'masked_tokens':len(ids)-len(expected_targets),'terminal_token_id':ids[terminal_position],
                         'terminal_supervised':True,'no_truncation':True,
                         'boundary_tokens':[{'index':i,'token':tok.convert_ids_to_tokens(ids[i]),'label':labels[i]}
                                            for i in sorted(set(range(max(0,expected_targets[0]-4),min(len(ids),expected_targets[0]+4))) |
                                                            set(range(max(0,expected_targets[-1]-2),len(ids))))]})
    batch = trainer.data_collator([dict(p) for p in trainer.train_dataset])
    padding = batch['attention_mask'] == 0
    assert (batch['labels'][padding] == -100).all()
    assert all((batch['labels'][i] != -100).sum().item() == e['supervised_tokens'] for i,e in enumerate(evidence))
    return {'rows':evidence,'full_tokens':sum(e['tokens'] for e in evidence),
            'supervised_tokens':sum(e['supervised_tokens'] for e in evidence),'padding_tokens':int(padding.sum()),
            'all_padding_masked':True,'template_sha256':hashlib.sha256(template.encode()).hexdigest(),
            'eos_token':tok.eos_token,'eos_token_id':tok.eos_token_id,'pad_token_id':tok.pad_token_id,
            'loss_type':trainer.args.loss_type,'labels_unshifted':True}


def finite_adapter_gradients(model):
    import torch
    nonzero, count, norm2 = False, 0, 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            assert param.grad is None, f"frozen parameter received gradients: {name}"
            continue
        if param.grad is not None:
            grad = param.grad.detach().float()
            assert torch.isfinite(grad).all(), f"nonfinite gradient: {name}"
            value = float(grad.square().sum())
            norm2 += value
            nonzero |= value > 0
            count += 1
    assert count and nonzero, "no genuine nonzero adapter gradient"
    return {'tensors_with_grad':count,'finite':True,'nonzero':nonzero,'norm_after_clipping':norm2**0.5,
            'frozen_base_gradients_absent':True}


def matching_inference_inputs(base_tokenizer, saved_tokenizer, messages):
    """Request an explicit BatchEncoding shape across tokenizer API versions."""
    import torch
    for attr in ('chat_template', 'eos_token_id', 'pad_token_id'):
        assert getattr(base_tokenizer, attr) == getattr(saved_tokenizer, attr), f"saved tokenizer changed {attr}"
    kwargs = dict(tokenize=True, add_generation_prompt=True, return_tensors='pt', return_dict=True)
    base = base_tokenizer.apply_chat_template(messages, **kwargs)
    saved = saved_tokenizer.apply_chat_template(messages, **kwargs)
    assert torch.equal(base['input_ids'], saved['input_ids']), 'saved tokenizer changed inference prompt IDs'
    assert torch.equal(base['attention_mask'], saved['attention_mask']), 'saved tokenizer changed attention mask'
    return base
