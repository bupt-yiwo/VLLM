import json
import os

def write_chat_template(processor, output_dir):
    output_chat_template_file = os.path.join(output_dir, "chat_template.json")
    chat_template_json_string = (
        json.dumps({"chat_template": processor.chat_template}, indent=2, sort_keys=True)
        + "\n"
    )
    with open(output_chat_template_file, "w", encoding="utf-8") as writer:
        writer.write(chat_template_json_string)

def save_all(model,processor,output_dir):
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    write_chat_template(processor, output_dir)