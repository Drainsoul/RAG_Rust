from transformers import AutoProcessor, AutoModelForMultimodalLM

processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
target_model = AutoModelForMultimodalLM.from_pretrained("google/gemma-4-E4B-it")

from fastapi import FastAPI
app = FastAPI()

@app.get("/{prompt}")
async def root(prompt: str):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True, 
    )
    inputs = processor(text=text, return_tensors="pt").to(target_model.device)
    input_len = inputs["input_ids"].shape[-1]
    outputs = target_model.generate(
        **inputs,
        max_new_tokens=256,
    )
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    out = processor.parse_response(response, prefix=text)

    print(out)
    return out['content']