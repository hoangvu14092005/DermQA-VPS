import logging
import warnings

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

# Monkeypatch to fix ImportError: cannot import name 'is_flash_attn_2_available' from 'transformers.modeling_utils' in newer transformers versions
import transformers.modeling_utils
try:
    from transformers.utils import is_flash_attn_2_available
except ImportError:
    try:
        from transformers.utils.import_utils import is_flash_attn_2_available
    except ImportError:
        def is_flash_attn_2_available():
            try:
                import flash_attn
                return True
            except ImportError:
                return False
transformers.modeling_utils.is_flash_attn_2_available = is_flash_attn_2_available

def dequantize_weight_param(weight_param):
    dequantized_weight = None
    
    # 1. Try peft helper with custom BNBState
    try:
        from peft.utils.integrations import dequantize_bnb_weight
        class BNBState:
            def __init__(self):
                self.SCB = None
        state = BNBState()
        try:
            dequantized_weight = dequantize_bnb_weight(weight_param, state=state, dtype=torch.bfloat16)
        except TypeError:
            dequantized_weight = dequantize_bnb_weight(weight_param, state=state)
    except Exception:
        pass

    # 2. Try transformers helper with custom BNBState
    if dequantized_weight is None:
        try:
            from transformers.integrations.bitsandbytes import dequantize_bnb_weight
            class BNBState:
                def __init__(self):
                    self.SCB = None
            state = BNBState()
            try:
                dequantized_weight = dequantize_bnb_weight(weight_param, state=state, dtype=torch.bfloat16)
            except TypeError:
                dequantized_weight = dequantize_bnb_weight(weight_param, state=state)
        except Exception:
            pass
            
    # 3. Manual Fallback
    if dequantized_weight is None:
        if type(weight_param).__name__ in ("Int8Params", "Params8bit") or weight_param.dtype == torch.int8:
            if hasattr(weight_param, "SCB") and weight_param.SCB is not None:
                try:
                    import bitsandbytes.functional as F
                    if hasattr(F, "int8_vectorwise_dequant"):
                        dequantized_weight = F.int8_vectorwise_dequant(weight_param.data, weight_param.SCB).to(torch.bfloat16)
                except Exception:
                    pass
                if dequantized_weight is None:
                    SCB = weight_param.SCB
                    dequantized_weight = (weight_param.data.to(torch.float32) * SCB.to(torch.float32).view(-1, 1)) * 7.874015718698502e-3
                    dequantized_weight = dequantized_weight.to(torch.bfloat16)
        elif type(weight_param).__name__ == "Params4bit":
            state = getattr(weight_param, "quant_state", None) or getattr(weight_param, "state", None)
            if state is not None:
                import bitsandbytes.functional as F
                dequantized_weight = F.dequantize_4bit(weight_param.data, state).to(torch.bfloat16)
                
    return dequantized_weight


def dequantize_and_replace_kv_b_proj(model):
    import torch.nn as nn
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, "kv_b_proj") and module.kv_b_proj is not None:
            kv_b_proj = module.kv_b_proj
            weight_param = kv_b_proj.weight
            
            # Check if weight is quantized using class name or dtype
            is_quantized = False
            if (weight_param.dtype == torch.int8 or 
                type(weight_param).__name__ in ("Int8Params", "Params8bit", "Params4bit")):
                is_quantized = True
                
            if is_quantized:
                try:
                    # Force bitsandbytes initialization if needed
                    needs_init = False
                    if type(weight_param).__name__ in ("Int8Params", "Params8bit"):
                        if not hasattr(weight_param, "SCB") or weight_param.SCB is None:
                            needs_init = True
                    elif type(weight_param).__name__ == "Params4bit":
                        state = getattr(weight_param, "quant_state", None) or getattr(weight_param, "state", None)
                        if state is None:
                            needs_init = True
                            
                    if needs_init:
                        try:
                            dummy_input = torch.zeros(1, kv_b_proj.in_features, device=weight_param.device, dtype=torch.bfloat16)
                            kv_b_proj(dummy_input)
                            weight_param = kv_b_proj.weight
                        except Exception:
                            pass
                            
                    # Dequantize the weight
                    dequantized = dequantize_weight_param(weight_param)
                    if dequantized is not None:
                        new_linear = nn.Linear(kv_b_proj.in_features, kv_b_proj.out_features, bias=False)
                        new_linear.weight = nn.Parameter(dequantized.to(torch.bfloat16), requires_grad=False)
                        new_linear = new_linear.to(weight_param.device)
                        module.kv_b_proj = new_linear
                        count += 1
                except Exception as e:
                    warnings.warn(f"Failed to permanently dequantize and replace {name}.kv_b_proj: {e}")
                    
    logging.info(f"Permanently dequantized and replaced {count} kv_b_proj layers with standard bfloat16 Linear layers.")


def get_custom_instruction(question_text):
    import re
    # Check if it is a Multiple Choice question (contains options like A., B., C., D.)
    has_mcq_options = (
        ("A." in question_text and "B." in question_text) or
        ("A. " in question_text and "B. " in question_text) or
        bool(re.search(r'\b[A-D]\.', question_text))
    )
    if has_mcq_options:
        return (
            "\nTrả lời bằng cách chỉ ghi ra (các) chữ cái đại diện cho đáp án đúng (ví dụ: A, B, hoặc AD). "
            "KHÔNG viết thêm bất kỳ giải thích hay từ ngữ nào khác."
        )
        
    # Check if it is a Judgement (Yes/No) question
    is_judgement = (
        "không?" in question_text or 
        "phải là" in question_text or
        "đúng không" in question_text
    )
    if is_judgement:
        return (
            "\nTrả lời bằng 'Có' hoặc 'Không' một cách trực tiếp. "
            "KHÔNG viết thêm bất kỳ giải thích hay từ ngữ nào khác."
        )
        
    # Default for short answers, fill-in-the-blank, and open-ended descriptions
    return (
        "\nTrả lời một cách trực tiếp, đầy đủ và ngắn gọn bằng tiếng Việt (không quá 2 câu, dưới 40 từ). "
        "KHÔNG viết các câu dẫn dắt mở đầu như 'Trong hình ảnh...' hoặc 'Dựa vào hình ảnh...'. "
        "KHÔNG chia danh sách hay dùng gạch đầu dòng."
    )


from .base import BaseModel


class DeepSeekVL2(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def check_install(self):
        try:
            import deepseek_vl2  # noqa: F401
        except Exception as e:
            logging.critical(
                'Please first install deepseek_vl2 from source codes in: https://github.com/deepseek-ai/DeepSeek-VL2')
            raise e

    def __init__(self, model_path='deepseek-ai/deepseek-vl2-tiny', load_in_4bit=False, load_in_8bit=False, **kwargs):
        self.check_install()

        pass

        assert model_path is not None
        self.model_path = model_path
        from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor

        self.vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["device_map"] = {"": torch.cuda.current_device()}
        elif load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = {"": torch.cuda.current_device()}

        model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if not (load_in_4bit or load_in_8bit):
            model = model.cuda()
        self.model = model.eval()
        
        if load_in_4bit or load_in_8bit:
            dequantize_and_replace_kv_b_proj(self.model)

        torch.cuda.empty_cache()
        default_kwargs = dict(max_new_tokens=128, repetition_penalty=1.1, do_sample=False, use_cache=True)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def prepare_inputs(self, message, dataset=None):

        if dataset == 'MMMU_DEV_VAL':

            def prepare_itlist(msgs):
                content, images = '', []
                image_idx = 1
                for s in msgs:
                    if s['type'] == 'image':
                        images.append(s['value'])
                        content += f'<image {image_idx}>'
                        image_idx += 1
                    elif s['type'] == 'text':
                        content += s['value']
                # content = '<image>' * (image_idx-1) + '\n' + content
                content = '<image>' * (image_idx - 1) + '\n' + content
                return content, images

            conversation = []
            if 'role' not in message[0]:
                content, images = prepare_itlist(message)
                content = content.replace(
                    'Please select the correct answer from the options above.',
                    "Answer with the option's letter from the given choices directly. Answer the question using a single word or phrase.\n"  # noqa
                )
                content = content.replace('Question:', "")
                content = content.replace('Options:\n', "")
                conversation.append(dict(role='<|User|>', content=content, images=images))
            else:
                role_map = {'user': '<|User|>', 'assistant': '<|Assistant|>'}
                for msgs in message:
                    role = role_map[msgs['role']]
                    content, images = prepare_itlist(msgs['content'])
                    content = content.replace(
                        'Please select the correct answer from the options above.',
                        "Answer with the option's letter from the given choices directly. Answer the question using a single word or phrase.\n"  # noqa
                    )
                    content = content.replace('Question:', "")
                    content = content.replace('Options:\n', "")
                    conversation.append(dict(role=role, content=content, images=images))
            conversation.append(dict(role='<|Assistant|>', content=''))

        else:

            def prepare_itlist(msgs):
                content, images = '', []
                for s in msgs:
                    if s['type'] == 'image':
                        images.append(s['value'])
                        content += '<image>\n'
                    elif s['type'] == 'text':
                        content += s['value']
                return content, images

            conversation = []
            if 'role' not in message[0]:
                content, images = prepare_itlist(message)
                instruction = get_custom_instruction(content)
                content += instruction
                conversation.append(dict(role='<|User|>', content=content, images=images))
            else:
                role_map = {'user': '<|User|>', 'assistant': '<|Assistant|>'}
                for i, msgs in enumerate(message):
                    role = role_map[msgs['role']]
                    content, images = prepare_itlist(msgs['content'])
                    if i == len(message) - 1 and role == '<|User|>':
                        instruction = get_custom_instruction(content)
                        content += instruction
                    conversation.append(dict(role=role, content=content, images=images))
            conversation.append(dict(role='<|Assistant|>', content=''))

        return conversation

    def generate_inner(self, message, dataset=None):
        conversation = self.prepare_inputs(message, dataset)
        from deepseek_vl2.utils.io import load_pil_images
        pil_images = load_pil_images(conversation)

        if dataset == 'MMMU_DEV_VAL':
            if len(pil_images):
                h, w = pil_images[0].size
                pil_images[0] = pil_images[0].resize((2 * h, 2 * w), Image.BILINEAR)

        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=(
                "You are a professional medical assistant specialized in dermatology. "
                "Answer the user's question about the skin lesion image in a direct, complete, and concise manner in the same language as the question. "
                "Follow these strict rules:\n"
                "1. Provide a complete, grammatically correct answer (do NOT output only keywords or a few words).\n"
                "2. Start answering directly. Do NOT write introductory sentences, conversational filler, or greetings.\n"
                "3. Write the answer as a single, coherent paragraph. Do NOT use bullet points or list formats.\n"
                "4. Keep the entire answer to 1 to 3 sentences (under 50 words) so that it is complete yet concise."
            )
        )
        prepare_inputs = prepare_inputs.to(self.model.device)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        inputs_embeds, past_key_values = self.model.incremental_prefilling(
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            chunk_size=512
        )

        # run the model to get the response
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            past_key_values=past_key_values,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.kwargs
        )

        answer = self.tokenizer.decode(
            outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(),
            skip_special_tokens=True
        )
        answer = answer.rstrip('.')

        return answer

    def chat_inner(self, message, dataset=None):
        return self.generate_inner(message, dataset=dataset)
