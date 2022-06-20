import argparse
import json
import logging
import types
import fnmatch

import torch

from lm_eval import tasks, evaluator
from lm_eval.base import BaseLM

import src.model
from src.model import RWKV_GPT, RWKV_RNN

class RWKV_LM(BaseLM):
    def __init__(self, model_name, device='cuda', batch_size=1):
        super().__init__()
        src.model.RUN_DEVICE = device
        self.model_name = model_name
        self.tokenizer = src.model.PreTrainedTokenizerFast(tokenizer_file='20B_tokenizer.json')
        self._batch_size = batch_size
        self._device = device
    def __str__(self):
        return self.__class__.__name__ + '/' + self.model_name
    @property
    def batch_size(self):
        return self._batch_size
    @property
    def device(self):
        return self._device
    def tok_decode(self, *params, **kwparams):
        return self.tokenizer.decode(*params, **kwparams)
    def tok_encode(self, *params, **kwparams):
        return self.tokenizer.encode(*params, **kwparams)
    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id
    @property
    def max_length(self):
        return src.model.ctx_len
    @property
    def max_gen_toks(self):
        return 256

class RWKV_model_GPT_FULL_LM(RWKV_LM):
    def __init__(self, model_name, device='cuda', batch_size=1):
        super().__init__(model_name, device, batch_size)
        self.model = RWKV_GPT(MODEL_NAME=model_name)
    def _model_call(self, inps):
        inps = inps.to(self.device)
        with torch.no_grad():
            self.model.clear()
            return self.model(inps)#[:,-1,:]
    def _model_generate(self, context, max_length, eos_token_id):
        context = context.to(self.device)
        start_len = context.shape[1]
        while context.shape[1] < max_length:
            self.model.clear()
            logits = self.model(context)[:,-1,:]
            token_ids = torch.argmax(logits,dim=-1)
            context = torch.cat([context, token_ids], dim=1)
        return context[:,start_len:]

class RWKV_model_GPT_RNN_LM(RWKV_model_GPT_FULL_LM):
    def __init__(self, model_name, device='cuda', batch_size=1):
        super().__init__(model_name, device, batch_size)
        self.model = RWKV_GPT(MODEL_NAME=model_name)
        self.rnn = RWKV_RNN(MODEL_NAME=model_name)
    def _model_generate(self, context, max_length, eos_token_id):
        self.model.clear()
        states = [types.SimpleNamespace() for idx in range(context.shape[0])]
        context = context.to(self.device)
        logits = self.model(context)[:,-1,:]
        self.model.save(states)
        output = torch.argmax(logits, dim=-1)
        while context.shape[1] < max_length:
            next_token_ids = []
            for state, batch in zip(states, output):
                self.rnn.load(state)
                logits_list = self.rnn.run([batch[-1]])
                self.rnn.save(state)
                token_id = max(range(self.tokenizer.vocab_size), key=lambda token_id: logits_list[token_id])
                next_token_ids.append(token_id)
            next_logits = torch.tensor(token_id, device=self.device)[None,:]
            output = torch.cat(output, next_logits, dim=1)
        return output
    

logging.getLogger("openai").setLevel(logging.WARNING)


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")

    return parser.parse_args()


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def main():
    args = parse_args()

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model={
            'GPT_FULL': RWKV_model_GPT_FULL_LM,
            'GPT_RNN': RWKV_model_GPT_RNN_LM
        }[args.model.split('/',1)[0]](args.model.split('/',1)[1], device=args.device, batch_size=args.batch_size),
        #model=RWKV_GPT(MODEL_NAME='all-10803', recurrent=False),
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
    )
    print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
