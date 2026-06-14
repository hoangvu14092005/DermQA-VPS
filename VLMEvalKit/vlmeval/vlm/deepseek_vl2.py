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

def wrap_attn_forward(attn_class):
    _orig_forward = attn_class.forward
    def patched_forward(self, *args, **kwargs):
        orig_weight = None
        if hasattr(self, "kv_b_proj") and hasattr(self.kv_b_proj, "weight"):
            weight_param = self.kv_b_proj.weight
            
            # Check if weight is quantized using class name or dtype
            is_quantized = False
            if (weight_param.dtype == torch.int8 or 
                type(weight_param).__name__ in ("Int8Params", "Params8bit", "Params4bit")):
                is_quantized = True
                
            if is_quantized:
                try:
                    dequantized_weight = None
                    # 1. Try peft helper
                    try:
                        from peft.utils.integrations import dequantize_bnb_weight
                        try:
                            dequantized_weight = dequantize_bnb_weight(weight_param, dtype=torch.bfloat16)
                        except TypeError:
                            dequantized_weight = dequantize_bnb_weight(weight_param)
                    except ImportError:
                        # 2. Try transformers helper
                        try:
                            from transformers.integrations.bitsandbytes import dequantize_bnb_weight
                            try:
                                dequantized_weight = dequantize_bnb_weight(weight_param, dtype=torch.bfloat16)
                            except TypeError:
                                dequantized_weight = dequantize_bnb_weight(weight_param)
                        except ImportError:
                            pass
                            
                    # 3. Manual Fallback
                    if dequantized_weight is None:
                        if hasattr(weight_param, "CB") and hasattr(weight_param, "SCB"):
                            CB = weight_param.CB
                            SCB = weight_param.SCB
                            dequantized_weight = (CB.to(torch.float32) * SCB.to(torch.float32).view(-1, 1)) / 127
                            dequantized_weight = dequantized_weight.to(torch.bfloat16)
                        elif hasattr(weight_param, "quant_state") or hasattr(weight_param, "state"):
                            state = getattr(weight_param, "quant_state", None) or getattr(weight_param, "state", None)
                            if state is not None:
                                import bitsandbytes.functional as F
                                dequantized_weight = F.dequantize_4bit(weight_param.data, state).to(torch.bfloat16)
                                
                    if dequantized_weight is not None:
                        orig_weight = self.kv_b_proj.weight
                        self.kv_b_proj.weight = dequantized_weight
                except Exception as e:
                    import warnings
                    warnings.warn(f"Failed to dequantize kv_b_proj.weight: {e}")
        try:
            return _orig_forward(self, *args, **kwargs)
        finally:
            if orig_weight is not None:
                self.kv_b_proj.weight = orig_weight
    attn_class.forward = patched_forward


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

        # Dynamically patch DeepseekV2Attention classes to dequantize kv_b_proj.weight on the fly when quantized
        if load_in_4bit or load_in_8bit:
            try:
                import deepseek_vl2.models.modeling_deepseek as modeling_deepseek
                for name in dir(modeling_deepseek):
                    obj = getattr(modeling_deepseek, name)
                    if isinstance(obj, type) and ("Attention" in name) and hasattr(obj, "forward"):
                        wrap_attn_forward(obj)
            except Exception as e:
                warnings.warn(f"Failed to apply DeepseekV2Attention monkeypatches: {e}")

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
                conversation.append(dict(role='<|User|>', content=content, images=images))
            else:
                role_map = {'user': '<|User|>', 'assistant': '<|Assistant|>'}
                for msgs in message:
                    role = role_map[msgs['role']]
                    content, images = prepare_itlist(msgs['content'])
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
            system_prompt=""
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
