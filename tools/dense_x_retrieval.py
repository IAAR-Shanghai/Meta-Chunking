import json
from nltk.tokenize import sent_tokenize

def dense_x_retrieval(tokenizer,model,text,title='',section='',target_size=256):
    title = title
    section = section
    target_size=target_size  # This model can only accept a maximum length of 512
    
    full_segments = sent_tokenize(text)
    merged_paragraphs = []  
    current_paragraph = "" 
    for paragraph in full_segments:  
        # Check if adding a new paragraph to the current paragraph exceeds the target size
        tmp_input_text = f"Title: {title}. Section: {section}. Content: {current_paragraph+' '+paragraph}"
        if len((tokenizer(tmp_input_text, return_tensors="pt").input_ids)[0].tolist()) <= target_size:  
            current_paragraph +=' '+paragraph  
        else:  
            merged_paragraphs.append(current_paragraph)  
            current_paragraph = paragraph  
    if current_paragraph:  
        merged_paragraphs.append(current_paragraph) 

    final_prop_list=[]
    for chunk in merged_paragraphs:
        content = chunk

        input_text = f"Title: {title}. Section: {section}. Content: {content}"
        print(input_text)
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        outputs = model.generate(input_ids.to(model.device), max_new_tokens=512).cpu()

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            prop_list = json.loads(output_text)
            final_prop_list=final_prop_list+prop_list
        except:
            print("[ERROR] Failed to parse output text as JSON.")
        # print(json.dumps(prop_list, indent=2))
    return final_prop_list