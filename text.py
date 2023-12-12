from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, fire, time
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"

REDACT_PROMPT = {
    "name": "Replace string representing any name in given text with [redacted].",
    "profession": "Replace string representing any profession, job name, job title in given text with [redacted].",
    "location": "Replace string representing any location, address, or place name in given text with [redacted].",
    "organization": "Replace string representing any organization, company, or group name in given text with [redacted].",
    "date": "Replace string representing any date, time, or duration in given text with [redacted].",
    "contact": "Replace string representing any phone number, email address, or other contact information in given text with [redacted].",
    "id": "Replace string representing any ID number, credit card number, or other personal identifier in given text with [redacted].",
    "이름" : "주어진 텍스트에서 이름을 나타내는 문자열을 [redacted]로 대체합니다.",
}

def replace_sequence(lst, sequence, replacement):
    sequence_length = len(sequence)
    i = 0
    while i <= len(lst) - sequence_length:
        # 지정된 수열이 현재 위치에서 시작하는지 확인합니다.
        if lst[i:i + sequence_length] == sequence:
            # 수열을 지정된 값으로 교체합니다.
            lst[i:i + sequence_length] = [replacement]
            # 교체 후, 수열 길이만큼 다음 위치로 이동합니다.
            i += sequence_length
        else:
            # 수열이 일치하지 않으면 다음 원소로 이동합니다.
            i += 1
    return lst

def remain_token(original_sentence, generated_sentence, redact_token_ids = [733, 893, 572, 286, 28793]):
    # print(generated_sentence.tolist()[0])
    length_of_input2 = len(replace_sequence(generated_sentence.tolist()[0], redact_token_ids, -1))
    # print(length_of_input2)
    # Trim the first 'length_of_input2' elements from input1 and return the result
    return original_sentence[:,length_of_input2-1:] # -1

def token_prediction_in_specified_index(input_ids, specified_ids):
    """
    ---
    >>> input_text = "Example input text"
    >>> input_ids = tokenizer.encode(input_text, return_tensors="pt")
    >>> token_prediction_in_specified_index(input_ids,input_ids)
    ' input'
    """
    # with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs.logits
    mask = torch.ones_like(predictions)*np.inf
    mask[:,:,specified_ids.tolist()[0]] = 0
    predictions-=mask
    predicted_index = torch.argmax(predictions[0, -1])
    predicted_word = tokenizer.decode([predicted_index.item()])
    predicted_index = torch.reshape(predicted_index, (1,1))
    return predicted_word, predicted_index

def redact_stream(want_to_redact, base_prompt):
    original_sentence =  base_prompt+want_to_redact+"</s>"
    generated_sentence = base_prompt
    original_sentence_token = tokenizer.encode(original_sentence, return_tensors='pt').to(device)
    generated_sentence_token = tokenizer.encode(generated_sentence, return_tensors='pt').to(device)
    print(generated_sentence.split(f"redacted text: ")[-1])
    while not generated_sentence.endswith(EOS) and generated_sentence_token.size()[1]-original_sentence_token.size()[1] < 1024:
        specified_ids = torch.cat([remain_token(original_sentence_token, generated_sentence_token) ,REDACTED_TOKEN ,SPACE_TOKEN, EOS_TOKEN], dim=1)
        predicted_word, next_token = token_prediction_in_specified_index(generated_sentence_token, specified_ids)
        generated_sentence_token = torch.cat([generated_sentence_token, next_token], dim=1) # token 추가
        generated_sentence = tokenizer.decode(generated_sentence_token.tolist()[0])
        for r in range(len(generated_sentence.split(f"redacted text: ")[-1].split("\n"))+1):
            print("\r", end="")
        print(generated_sentence.split(f"redacted text: ")[-1], end="")
        time.sleep(1)
    print("\n Redacted text: ", generated_sentence.split(f"### redacted text: ")[-1].removeprefix("<s>").removesuffix(EOS))

def main(want_to_redact = "My name is Mr. Kim.", redact_tag = "profession", base_prompt = ""):
    # print(want_to_redact)
    PROMPT_TEMPLATE= {
        "AIDC-ai-business/Marcoroni-7b-v3":f"### Instruction:\n\n{REDACT_PROMPT[redact_tag]}\n\ngiven text: {want_to_redact}</s>\n\n\n### Response: \n\nredacted text: ",
        "mistralai/Mixtral-8x7B-Instruct-v0.1":f"[INST] {REDACT_PROMPT[redact_tag]}\n\ngiven text: {want_to_redact}</s> [/INST] redacted text: ",
        "./trained_model":f"[INST] {REDACT_PROMPT[redact_tag]}\n\ngiven text: {want_to_redact}</s> [/INST] redacted text: ",
    }

    redact_prompt = REDACT_PROMPT[redact_tag]
    base_prompt = PROMPT_TEMPLATE[model_id].format(redact_prompt=redact_prompt, want_to_redact=want_to_redact)
    redact_stream(want_to_redact, base_prompt)

def redact(want_to_redact, base_prompt):
    original_sentence =  base_prompt+want_to_redact+"</s>"
    generated_sentence = base_prompt
    original_sentence_token = tokenizer.encode(original_sentence, return_tensors='pt').to(device)
    generated_sentence_token = tokenizer.encode(generated_sentence, return_tensors='pt').to(device)
    print(generated_sentence.split(f"redacted text: ")[-1])
    while not generated_sentence.endswith(EOS) and generated_sentence_token.size()[1]-original_sentence_token.size()[1] < 1024:
        specified_ids = torch.cat([remain_token(original_sentence_token, generated_sentence_token) ,REDACTED_TOKEN ,SPACE_TOKEN, EOS_TOKEN], dim=1)
        predicted_word, next_token = token_prediction_in_specified_index(generated_sentence_token, specified_ids)
        generated_sentence_token = torch.cat([generated_sentence_token, next_token], dim=1) # token 추가
        generated_sentence = tokenizer.decode(generated_sentence_token.tolist()[0])
        for r in range(len(generated_sentence.split(f"redacted text: ")[-1].split("\n"))+1):
            print("\r", end="")
        print(generated_sentence.split(f"redacted text: ")[-1], end="")
    return generated_sentence.split(f"redacted text: ")[-1].removeprefix("<s>").removesuffix(EOS)

def greet(want_to_redact):
    redact_tag = "이름"

    # print(want_to_redact)
    PROMPT_TEMPLATE= {
        "AIDC-ai-business/Marcoroni-7b-v3":f"### Instruction:\n\n{REDACT_PROMPT[redact_tag]}\n\ngiven text: {want_to_redact}</s>\n\n\n### Response: \n\nredacted text: ",
        "mistralai/Mixtral-8x7B-Instruct-v0.1":f"[INST] {REDACT_PROMPT[redact_tag]}\n\ngiven text: {want_to_redact}</s> [/INST] redacted text: ",
        "./trained_model":f"[INST] {REDACT_PROMPT[redact_tag]}\n\ngiven text: {want_to_redact}</s> [/INST] redacted text: ",
    }

    redact_prompt = REDACT_PROMPT[redact_tag]
    base_prompt = PROMPT_TEMPLATE[model_id].format(redact_prompt=redact_prompt, want_to_redact=want_to_redact)
    return redact(want_to_redact, base_prompt)

if __name__=="__main__":

    model_id = "AIDC-ai-business/Marcoroni-7b-v3" # "mistralai/Mixtral-8x7B-Instruct-v0.1" #
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = '/data', load_in_8bit=False, load_in_4bit=True) #use_flash_attention_2=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir = '/data', device_map="auto")
    REDACTED_TOKEN = tokenizer.encode("[redacted]", return_tensors='pt')[:,1:].to(device)
    SPACE_TOKEN = tokenizer.encode(" ", return_tensors='pt')[:,1:].to(device)
    EOS_TOKEN = torch.Tensor([[tokenizer.eos_token_id]]).to(device)
    EOS = tokenizer.decode(tokenizer.eos_token_id)

    # fire.Fire(main) 

    import gradio as gr

    demo = gr.Interface(fn=greet, inputs="text", outputs="text")
    demo.launch(share=True)   