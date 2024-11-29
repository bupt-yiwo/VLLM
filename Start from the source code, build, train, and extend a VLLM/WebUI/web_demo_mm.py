# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import re
from argparse import ArgumentParser
from threading import Thread
import gradio as gr
import torch
from qwen_vl_utils import process_vision_info
from transformers import TextIteratorStreamer
from model.modeling_qwen_llama import LlamaForCausalLM
from model.model_processing import LQ_Tokenizer

DEFAULT_CKPT_PATH = '/home/zhuyao/Sunpeng/llava_qwen/check_point/instruct_525k'
DEFAULT_P_PATH = "/home/zhuyao/Sunpeng/models/qwen_2B_instruct"
DEFAULT_T_PATH = '/home/zhuyao/Sunpeng/llava_qwen/tes'

def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-c',
                        '--checkpoint-path',
                        type=str,
                        default=DEFAULT_CKPT_PATH,
                        help='Checkpoint name or path, default to %(default)r')
    parser.add_argument('--processor_path',
                    type=str,
                    default=DEFAULT_P_PATH,
                    help='processor_path')
    parser.add_argument('--tokenizer_path',
                        type=str,
                    default=DEFAULT_T_PATH,
                    help='tokenizer_path')
    parser.add_argument('--cpu-only', action='store_true', help='Run demo with CPU only')

    parser.add_argument('--flash-attn2',
                        action='store_true',
                        default=False,
                        help='Enable flash_attention_2 when loading the model.')
    parser.add_argument('--share',
                        action='store_true',
                        default=False,
                        help='Create a publicly shareable link for the interface.')
    parser.add_argument('--inbrowser',
                        action='store_true',
                        default=False,
                        help='Automatically launch the interface in a new tab on the default browser.')
    parser.add_argument('--server-port', type=int, default=7860, help='Demo server port.')
    parser.add_argument('--server-name', type=str, default='127.0.0.1', help='Demo server name.')

    args = parser.parse_args()
    return args


def _load_model_processor(args):
    # if args.cpu_only:
    #     device_map = 'cpu'
    # else:
    #     device_map = 'auto'

    # Check if flash-attn2 flag is enabled and load model accordingly
    if args.flash_attn2:
        model = LlamaForCausalLM.from_pretrained(args.checkpoint_path,device_map = "cuda:7",torch_dtype='auto',attn_implementation="flash_attention_2")
    else:
        model = LlamaForCausalLM.from_pretrained(args.checkpoint_path,device_map = "cuda:7",torch_dtype='auto')
  
    min_image_tokens = 4
    max_image_tokens = 336
    processor = LQ_Tokenizer(args.tokenizer_path,args.processor_path,min_image_tokens,max_image_tokens)
    return model, processor


def _parse_text(text):
    lines = text.split('\n')
    lines = [line for line in lines if line != '']
    count = 0
    for i, line in enumerate(lines):
        if '```' in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = '<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace('`', r'\`')
                    line = line.replace('<', '&lt;')
                    line = line.replace('>', '&gt;')
                    line = line.replace(' ', '&nbsp;')
                    line = line.replace('*', '&ast;')
                    line = line.replace('_', '&lowbar;')
                    line = line.replace('-', '&#45;')
                    line = line.replace('.', '&#46;')
                    line = line.replace('!', '&#33;')
                    line = line.replace('(', '&#40;')
                    line = line.replace(')', '&#41;')
                    line = line.replace('$', '&#36;')
                lines[i] = '<br>' + line
    text = ''.join(lines)
    return text


def _remove_image_special(text):
    text = text.replace('<ref>', '').replace('</ref>', '')
    return re.sub(r'<box>.*?(</box>|$)', '', text)


def _is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _transform_messages(original_messages):
    transformed_messages = []
    for message in original_messages:
        new_content = []
        for item in message['content']:
            if 'image' in item:
                new_item = {'type': 'image', 'image': item['image']}
            elif 'text' in item:
                new_item = {'type': 'text', 'text': item['text']}
            elif 'video' in item:
                new_item = {'type': 'video', 'video': item['video']}
            else:
                continue
            new_content.append(new_item)

        new_message = {'role': message['role'], 'content': new_content}
        transformed_messages.append(new_message)

    return transformed_messages


def _launch_demo(args, model, processor):

    def call_local_model(model, processor, messages):

        messages = _transform_messages(messages)

        # text = processor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor([messages])
        inputs = {k:v.to(model.device) for k,v in inputs.items()}
        generetion_config = {
            "bos_token_id": processor.tokenizer.bos_token_id,
            "do_sample": True,
            "eos_token_id": [128009],
            "pad_token_id":processor.tokenizer.pad_token_id,
            "repetition_penalty": 1.15,
            "temperature": 1.0,
            "top_p": 0.001,
            "top_k": 5,
            "transformers_version": "4.37.0"
        }
        tokenizer = processor.tokenizer
        streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {'max_new_tokens': 1000, 
                      'streamer': streamer, 
                      **inputs,
                      **generetion_config}

        thread = Thread(target=model.generate,kwargs=gen_kwargs)
        thread.start()

        generated_text = ''
        for new_text in streamer:
            generated_text += new_text
            yield generated_text

    def create_predict_fn():

        def predict(_chatbot, task_history):
            nonlocal model, processor
            chat_query = _chatbot[-1][0]
            query = task_history[-1][0]
            if len(chat_query) == 0:
                _chatbot.pop()
                task_history.pop()
                return _chatbot
            print('User: ' + _parse_text(query))
            history_cp = copy.deepcopy(task_history)
            full_response = ''
            messages = []
            content = []
            for q, a in history_cp:
                if isinstance(q, (tuple, list)):
                    if _is_video_file(q[0]):
                        content.append({'video': f'file://{q[0]}'})
                    else:
                        content.append({'image': f'file://{q[0]}'})
                else:
                    content.append({'text': q})
                    messages.append({'role': 'user', 'content': content})
                    messages.append({'role': 'assistant', 'content': [{'text': a}]})
                    content = []
            messages.pop()

            for response in call_local_model(model, processor, messages):
                _chatbot[-1] = (_parse_text(chat_query), _remove_image_special(_parse_text(response)))

                yield _chatbot
                full_response = _parse_text(response)

            task_history[-1] = (query, full_response)
            print('Qwen-VL-Chat: ' + _parse_text(full_response))
            yield _chatbot

        return predict

    def create_regenerate_fn():

        def regenerate(_chatbot, task_history):
            nonlocal model, processor
            if not task_history:
                return _chatbot
            item = task_history[-1]
            if item[1] is None:
                return _chatbot
            task_history[-1] = (item[0], None)
            chatbot_item = _chatbot.pop(-1)
            if chatbot_item[0] is None:
                _chatbot[-1] = (_chatbot[-1][0], None)
            else:
                _chatbot.append((chatbot_item[0], None))
            _chatbot_gen = predict(_chatbot, task_history)
            for _chatbot in _chatbot_gen:
                yield _chatbot

        return regenerate

    predict = create_predict_fn()
    regenerate = create_regenerate_fn()

    def add_text(history, task_history, text):
        task_text = text
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(task_text, None)]
        return history, task_history, ''

    def add_file(history, task_history, file):
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]
        return history, task_history

    def reset_user_input():
        return gr.update(value='')

    def reset_state(_chatbot, task_history):
        task_history.clear()
        _chatbot.clear()
        _gc()
        return []

    with gr.Blocks() as demo:
        import base64
#         gr.Markdown("""\
# <p align="center"><img src="https://s2.loli.net/2024/11/28/t8iFOpHhoGKIje4.png" style="height: 160px"/><p>"""
#                    )
        with open("/home/zhuyao/Sunpeng/274e3ef7c2bc430877b5e15d2838e1c1.jpg", "rb") as img_file:
            b64_string = base64.b64encode(img_file.read()).decode("utf-8")

        html = f'<p align="center"><img src="data:image/png;base64,{b64_string}" style="height: 160px"/></p>'
        gr.Markdown(html)
        gr.Markdown("""<center><font size=8>SP-MODEL</center>""")

        chatbot = gr.Chatbot(label='SP-MODEL', elem_classes='control-height', height=600)
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])

        with gr.Row():
            addfile_btn = gr.UploadButton('📁 Upload (上传文件)', file_types=['image', 'video'])
            submit_btn = gr.Button('🚀 Submit (发送)')
            regen_btn = gr.Button('🤔️ Regenerate (重试)')
            empty_bin = gr.Button('🧹 Clear History (清除历史)')

        submit_btn.click(add_text, [chatbot, task_history, query],
                         [chatbot, task_history]).then(predict, [chatbot, task_history], [chatbot], show_progress=True)
        submit_btn.click(reset_user_input, [], [query])
        empty_bin.click(reset_state, [chatbot, task_history], [chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)


    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()
    model, processor = _load_model_processor(args)
    _launch_demo(args, model, processor)


if __name__ == '__main__':
    main()

