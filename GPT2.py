import sys
import random
import argparse

import torch
import torch.nn.functional as F
import numpy as np

from transformers import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config,)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
}


def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, tokenizer, length, context, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    device='cpu'):
    text = ''
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(1, 1)
    generated = context
    with torch.no_grad():
        for _ in range(length):

            inputs = {'input_ids': generated}

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(1):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty
                
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            
            out = generated[:, context.shape[1]:].tolist()
            chunk = tokenizer.decode(out[0], clean_up_tokenization_spaces=True)[len(text):]
            text += chunk
            if text.find('<|endoftext|>') != -1:
                break
            print(chunk, end='')
            sys.stdout.flush()
    return
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=" ")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument('--seed', type=int, default=random.getrandbits(32),
                        help="random seed for initialization")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature of 0 implies greedy sampling")
    args = parser.parse_args()
    
    model_type = 'gpt2'
    model_name_or_path = 'gpt2-xl'
    prompt = args.prompt
    length = args.length
    temperature = args.temperature
    repetition_penalty = 1.0
    top_k = 0
    top_p = 0.9
    no_cuda = False
    seed = args.seed

    try:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()

        set_seed(seed, n_gpu)

        model_type = model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[model_type]
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
        model = model_class.from_pretrained(model_name_or_path)
        model.to(device)
        model.eval()

        if length < 0 and model.config.max_position_embeddings > 0:
            length = model.config.max_position_embeddings
        elif 0 < model.config.max_position_embeddings < length:
            length = model.config.max_position_embeddings  # No generation bigger than model size 
        elif length < 0:
            length = MAX_LENGTH  # avoid infinite loop

        raw_text = prompt
        context_tokens = tokenizer.encode(raw_text, add_special_tokens=False)
        sample_sequence(
            model=model,
            tokenizer=tokenizer,
            context=context_tokens,
            length=length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=device,
        )
    except:
        print('Please try again when I am less busy...');
    return
    

if __name__ == '__main__':
    main()