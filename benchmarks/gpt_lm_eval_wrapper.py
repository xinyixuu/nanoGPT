# gpt_lm_eval_wrapper.py

import torch
import torch.nn.functional as F
import lm_eval.api.model as model_api

class NanoGPTLM(model_api.LM):
    def __init__(self, model, tokenizer_encode, tokenizer_decode, eot_token_id, device, max_new_tokens, batch_size=1, temperature=1.0, top_k=None, stop_string=None):
        super().__init__()
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self._eot_token_id = eot_token_id
        self.device = device
        self._batch_size = batch_size
        self.temperature = temperature
        self.top_k = top_k
        self.stop_string = stop_string
        self.max_new_tokens = max_new_tokens

    @property
    def eot_token_id(self):
        return self._eot_token_id

    @property
    def max_length(self):
        return self.model.config.block_size

    @property
    def batch_size(self):
        return self._batch_size

    def tok_encode(self, string):
        return self.tokenizer_encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer_decode(tokens)

    def _model_call(self, inps):
        logits, _ = self.model(inps)
        return logits

    def _model_generate(self, context, max_length, eos_token_id):
        idx = context.to(self.device)
        idx = self.model.generate(
            idx, 
            max_new_tokens=max_length, 
            temperature=self.temperature, 
            top_k=self.top_k
        )
        return idx

    def loglikelihood(self, requests):
        res = []
        batch_size = self.batch_size

        for batch_start in range(0, len(requests), batch_size):
            batch = requests[batch_start:batch_start+batch_size]
            contexts, continuations = zip(*batch)

            context_tokens = [self.tok_encode(c) for c in contexts]
            continuation_tokens = [self.tok_encode(c) for c in continuations]

            max_len = max(len(ctx) + len(cont) for ctx, cont in zip(context_tokens, continuation_tokens))
            batch_tokens = []
            for ctx, cont in zip(context_tokens, continuation_tokens):
                tokens = ctx + cont
                pad_len = max_len - len(tokens)
                tokens += [self.eot_token_id] * pad_len
                batch_tokens.append(tokens)

            batch_tokens = torch.tensor(batch_tokens, device=self.device)

            with torch.no_grad():
                logits, _ = self.model(batch_tokens[:, :-1])
                log_probs = F.log_softmax(logits, dim=-1)

                for i, (ctx, cont) in enumerate(zip(context_tokens, continuation_tokens)):
                    ctx_len = len(ctx)
                    cont_len = len(cont)

                    logits_for_cont = log_probs[i, ctx_len-1:ctx_len-1+cont_len]
                    targets = batch_tokens[i, ctx_len:ctx_len+cont_len]

                    selected_log_probs = logits_for_cont.gather(1, targets.unsqueeze(-1)).squeeze(-1)
                    ll = selected_log_probs.sum().item()
                    res.append((ll, True))

        return res

    def loglikelihood_rolling(self, requests):
        res = []
        batch_size = self.batch_size

        for batch_start in range(0, len(requests), batch_size):
            batch = requests[batch_start:batch_start+batch_size]
            all_tokens = [self.tok_encode(r) for r in batch]
            max_len = max(len(t) for t in all_tokens)

            input_batch = []
            for tokens in all_tokens:
                pad_len = max_len - len(tokens)
                tokens += [self.eot_token_id] * pad_len
                input_batch.append(tokens)

            input_batch = torch.tensor(input_batch, device=self.device)

            with torch.no_grad():
                logits, _ = self.model(input_batch[:, :-1])
                log_probs = F.log_softmax(logits, dim=-1)

                for i, tokens in enumerate(all_tokens):
                    tokens_tensor = torch.tensor(tokens, device=self.device)

                    logits_for_tokens = log_probs[i, :tokens_tensor.shape[0]-1]
                    targets = tokens_tensor[1:]

                    selected_log_probs = logits_for_tokens.gather(1, targets.unsqueeze(-1)).squeeze(-1)
                    ll = selected_log_probs.sum().item()
                    res.append(ll)

        return res

    def generate_until(self, requests):
        res = []
        for context, until in requests:
            context_tokens = torch.tensor([self.tok_encode(context)], device=self.device)
            idx, generated_text = self.model.generate_with_stop(
                context_tokens,
                max_new_tokens=self.max_new_tokens,
                # assuming until is a list of stop strings
                stop_string=until[0],
                decode=self.tokenizer_decode,
                temperature=self.temperature,
                top_k=self.top_k
            )
            new_text = generated_text[len(context):]
            res.append(new_text)
        return res
