# Code to inference Open Hermes 2.5 with HF Transformers
# Requires pytorch, transformers, bitsandbytes, sentencepiece, protobuf, and flash-attn packages

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM, MistralForCausalLM
import bitsandbytes, flash_attn
import os

class NERSystem:
    def __init__(model_path):
        print(model_path)
        if not os.path.exists(model_path):
            raise Exception("Model path does not exist")
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = MistralForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",#{'': 'cuda:0'},
                    load_in_8bit=False,
                    load_in_4bit=True,
                    use_flash_attention_2=True
                )

    def ner_inference(self, texts):

        prompts = [
            """<|im_start|>system
            You are a superintelligent artificial general intelligence, here to teach and assist me.<|im_end|>
            <|im_start|>user
            Replace with [redacted] any strings that might be the name, abbreviations, initials, or the person's first name.
            """+text+"""<|im_end|>
            <|im_start|>assistant""" for text in texts]
        for chat in prompts:
            print(chat)
            input_ids = tokenizer(chat, return_tensors="pt").input_ids.to("cuda")
            generated_ids = model.generate(input_ids, max_new_tokens=750, temperature=0.8, repetition_penalty=1.1, do_sample=True, eos_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True)
            print(f"Response: {response}")

        return response

if __name__ == "__main__":
    ner_system = NERSystem(model_path = "./trained_model")
    texts = ["hello, I'm John Smith"]
    print(ner_system.ner_inference(texts))