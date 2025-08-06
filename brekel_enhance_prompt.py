#
# Brekel Prompt Enhancer Node for ComfyUI
# Version: 1.1.0
#
# Author: Brekel - https://brekel.com
#
# This custom node provides a generic Large Language Model (LLM) based prompt enhancer.
#
# Key Features:
# - Dynamically detects and lists available LLMs from your ComfyUI/models/LLM directory.
# - Loads system prompts dynamically from .txt files in a 'ComfyUI/custom_nodes/ComfyUI-Brekel/prompt_enhancer' subfolder.
# - Supports 16-bit precision, 8-bit, and 4-bit quantization for VRAM efficiency.
# - Logs the model's memory footprint upon loading.
# - Defaults to SDPA (Scaled Dot-Product Attention) for optimal memory and speed.
# - Flexible memory management with performance reporting for offloading.
# - Controllable creativity, seed, and target_length for prompt generation.
#
#
# Release log:
#
# v1.1.0:
# - added a fifth file selector
# - node now shows the prompt it generated on the node itself after it has run
# - cleaned up the length_instruction that get's added to the system prompt a bit


# --- CONFIGURATION CONSTANT ---
# Define the subfolder name where custom system prompt text files are stored.
# This folder is expected to be inside the custom node's directory.
SUBFOLDER_NAME = "prompt_enhancer"


import os
import torch
import logging
import time
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# import block to communicate with the frontend
try:
    from server import PromptServer
except ImportError:
    # If the server is not available (e.g., in a headless environment)  create a dummy class to prevent errors.
    class PromptServer:
        instance = None


# --- Setup basic logging ---
# Configures a logger to output informational messages to the console.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Global variables for Caching ---
# These globals cache the loaded model and tokenizer to avoid reloading them on every execution.
# This significantly speeds up consecutive runs with the same model settings.
_loaded_model_name = None       # Stores the name of the currently loaded model.
_loaded_quantization = None     # Stores the quantization setting of the loaded model.
_loaded_model = None            # Caches the loaded model object itself.
_loaded_tokenizer = None        # Caches the loaded tokenizer object.
_model_is_offloaded = False     # Flag to track if the model is currently offloaded to CPU RAM.


# --- HELPER FUNCTIONS ---
def get_comfy_root_dir():
    """
    Traverses up the directory tree to find the ComfyUI root directory.
    This is essential for reliably locating the 'models/LLM' folder,
    regardless of how the user has installed ComfyUI or the custom node.
    It identifies the root by looking for the 'main.py' file.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        # Check if the current directory contains 'main.py' and is named 'comfyui'
        if os.path.exists(os.path.join(current_dir, "main.py")) and os.path.basename(current_dir).lower() == "comfyui":
            return current_dir
        # Move up one directory
        parent_dir = os.path.dirname(current_dir)
        # If we have reached the filesystem root, stop and return a fallback path.
        if parent_dir == current_dir:
            # Fallback for unusual directory structures.
            return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        current_dir = parent_dir


def _format_bytes(size_in_bytes: int) -> str:
    """Formats a byte value into a human-readable string (MB or GB)."""
    if size_in_bytes >= 1024**3:
        return f"{size_in_bytes / (1024**3):.2f} GB"
    return f"{size_in_bytes / (1024**2):.2f} MB"


def get_llm_models():
    """
    Scans the 'ComfyUI/models/LLM' directory for available Hugging Face models.
    It identifies a model by the presence of a 'config.json' file in a subdirectory.
    Returns a sorted list of model names for the dropdown menu.
    """
    try:
        llm_dir = os.path.join(get_comfy_root_dir(), "models", "LLM")
        if not os.path.isdir(llm_dir):
            return ["NO_MODEL_FOUND"]
        # A valid model is a directory containing a config.json file.
        models = [d for d in os.listdir(llm_dir) if os.path.isdir(os.path.join(llm_dir, d)) and os.path.exists(os.path.join(llm_dir, d, "config.json"))]
        if not models:
            return ["NO_MODEL_FOUND"]
        
        # Sort models alphabetically but move a preferred default to the top if it exists.
        sorted_models = sorted(models)
        default_model = "Llama-3.2-1B-Instruct"
        if default_model in sorted_models:
            sorted_models.insert(0, sorted_models.pop(sorted_models.index(default_model)))
        return sorted_models
    except Exception as e:
        logger.error(f"Could not scan for LLM models: {e}")
        return ["ERROR_SCANNING_MODELS"]


def get_system_prompts():
    """
    Dynamically loads system prompts from .txt files in the 'prompt_enhancer' subfolder.
    The filename (without extension) becomes the prompt's display name in the UI.
    Returns a dictionary mapping display names to their text content.
    """
    prompts_dir = os.path.join(os.path.dirname(__file__), SUBFOLDER_NAME)

    if not os.path.isdir(prompts_dir):
        error_msg = f"System prompts directory not found: {prompts_dir}. Please reinstall the node via the ComfyUI Manager."
        logger.error(error_msg)
        return {"ERROR": error_msg}

    loaded_prompts = {}
    try:
        for filename in sorted(os.listdir(prompts_dir)):
            if filename.endswith(".txt"):
                try:
                    with open(os.path.join(prompts_dir, filename), 'r', encoding='utf-8') as f:
                        # Create a user-friendly name from the filename.
                        display_name = os.path.splitext(filename)[0].replace('_', ' ').strip()
                        loaded_prompts[display_name] = f.read()
                except Exception as e:
                    logger.warning(f"Failed to read or decode prompt file {filename}: {e}")
        
        if not loaded_prompts:
            return {"NO PROMPTS FOUND": f"Error: No .txt files found in {prompts_dir}."}
        
        return loaded_prompts
    except Exception as e:
        error_msg = f"Fatal error reading system prompts from {prompts_dir}: {e}"
        logger.error(error_msg)
        return {"ERROR": error_msg}




# --- CUSTOM COMFYUI NODE CLASS ---
class BrekelEnhancePrompt:
    """
    The main class for the ComfyUI custom node. It defines the node's inputs,
    functionality, and how it integrates with the ComfyUI system.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input widgets that will appear on the node in the ComfyUI interface.
        This method is called by ComfyUI to build the node's UI.
        """
        system_prompt_names = list(get_system_prompts().keys())
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "a cat wearing a wizard hat", "tooltip": "Original prompt to enhance."}),
                "prefix": ("STRING", {"multiline": False, "default": "", "tooltip": "Prefix to prepend at the start of the prompt, for example to add your Lora trigger word(s)."}),
                "model_name": (get_llm_models(), {"tooltip": "LLM model to use for prompt enhancement."}),
                "quantization": (["Disabled", "8-bit (Int8)", "4-bit (NF4)"], {"default": "Disabled", "tooltip": "Quantization method to save VRAM."}),
                "memory_management": (["Offload to CPU after use", "Keep in VRAM", "Unload completely after use"], {"tooltip": "Memory management strategy."}),
                "system_prompt": (system_prompt_names, {"tooltip": "System prompt from files in the 'prompt_enhancer' subfolder."}),
                "target_length": ("INT", {"default": 150, "min": 64, "max": 512, "step": 8, "display": "slider", "tooltip": "Target length for the generated prompt. Used as a goal (characters) and a safe token limit."}),
                "creativity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "display": "slider", "tooltip": "Creativity level (temperature)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "postfix": ("STRING", {"multiline": False, "default": "", "tooltip": "Postfix to append at the end of the prompt, for example to add your Lora trigger word(s)."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def IS_CHANGED(cls, prompt, seed, **kwargs):
        """
        A method used by ComfyUI to determine if the node needs to be re-executed.
        If the seed is 0 (random), it always re-runs. Otherwise, it re-runs only
        if any of the inputs have changed.
        """
        if seed == 0:
            return time.time()  # Return current time to ensure it's always unique.
        # Hash all inputs to create a unique ID. If the ID changes, the node re-runs.
        all_inputs = [str(prompt), str(seed)] + [str(v) for k, v in sorted(kwargs.items())]
        return hash("".join(all_inputs))

    # --- Standard ComfyUI Node properties ---
    CATEGORY = "Brekel"
    FUNCTION = "brekel_enhance_prompt"
    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = False

    def _load_model(self, model_name: str, quantization: str):
        """
        Handles the loading, caching, and quantization of the LLM.
        This internal method is responsible for all model management logic.
        """
        global _loaded_model, _loaded_tokenizer, _loaded_model_name, _model_is_offloaded, _loaded_quantization

        # --- CACHE CHECK ---
        # If the requested model and quantization are already loaded, we might not need to do anything.
        if _loaded_model_name == model_name and _loaded_quantization == quantization:
            # If the model was offloaded to CPU, move it back to GPU.
            if _model_is_offloaded and _loaded_model is not None:
                if quantization == "Disabled" and _loaded_model.device.type == 'cpu':
                    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    logger.info(f"Moving model '{model_name}' from CPU back to {target_device}...")
                    start_time = time.perf_counter()
                    _loaded_model.to(target_device)
                    logger.info(f"Model moved to {target_device} in {time.perf_counter() - start_time:.2f} seconds.")
                else:
                    # Quantized models are managed by the `accelerate` library and don't need manual moving.
                    logger.info(f"Quantized model '{model_name}' is managed by accelerate. No move needed.")
                _model_is_offloaded = False
            return # Model is ready, exit the function.
        
        # --- UNLOAD PREVIOUS MODEL ---
        # If a different model is requested, unload the old one first.
        if _loaded_model is not None:
            logger.info(f"Unloading previous model '{_loaded_model_name}' to load new one.")
            del _loaded_model, _loaded_tokenizer
            torch.cuda.empty_cache()
            _loaded_model = _loaded_tokenizer = _loaded_model_name = _loaded_quantization = None

        # --- LOAD NEW MODEL ---
        models_base_dir = os.path.join(get_comfy_root_dir(), "models", "LLM")
        model_path = os.path.join(models_base_dir, model_name)
        logger.info(f"Loading '{model_name}' with quantization: {quantization}...")
        
        try:
            # Base arguments for loading the model. `device_map="auto"` lets `accelerate` handle device placement.
            # `attn_implementation="sdpa"` uses PyTorch's efficient attention mechanism.
            model_load_args = {"device_map": "auto", "attn_implementation": "sdpa"}
            quantization_config = None

            # --- CONFIGURE QUANTIZATION ---
            if quantization == "Disabled":
                # Use 16-bit floating point precision (bfloat16 if available, else float16).
                model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
                model_load_args["torch_dtype"] = model_dtype
                logger.info(f"Using 16-bit precision ({model_dtype}).")
            else:
                # Quantization requires specific libraries.
                try:
                    import bitsandbytes, accelerate
                except ImportError:
                    raise ImportError("Quantization requires 'bitsandbytes' and 'accelerate'. Please install them via: pip install -r custom_nodes/Brekel/requirements.txt")

                if quantization == "8-bit (Int8)":
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                elif quantization == "4-bit (NF4)":
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
                
                if quantization_config:
                    model_load_args["quantization_config"] = quantization_config
                    logger.info(f"Using {quantization} quantization via BitsAndBytesConfig.")

            # Load the model and tokenizer from the specified path with the configured arguments.
            logger.info("Using SDPA (PyTorch native attention) for optimal performance.")
            _loaded_model = AutoModelForCausalLM.from_pretrained(model_path, **model_load_args)
            _loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Log the memory usage for user information.
            try:
                footprint_bytes = _loaded_model.get_memory_footprint()
                logger.info(f"Model memory footprint: {_format_bytes(footprint_bytes)}")
                if hasattr(_loaded_model, 'hf_device_map'):
                    logger.info(f"Model device map: {_loaded_model.hf_device_map}")
            except Exception as e:
                logger.warning(f"Could not retrieve model memory footprint: {e}")

            # Update global cache variables with the newly loaded model's info.
            _loaded_model_name = model_name
            _loaded_quantization = quantization
            _model_is_offloaded = False
            logger.info(f"Successfully loaded '{model_name}'.")

        except Exception as e:
            # If loading fails, clear all cache variables and free memory.
            _loaded_model = _loaded_tokenizer = _loaded_model_name = _loaded_quantization = None
            torch.cuda.empty_cache()
            raise RuntimeError(f"Failed to load model '{model_name}'. Error: {e}")


    def _clean_llm_output(self, text: str) -> str:
        """
        Cleans the raw output from the LLM, removing common conversational boilerplate.
        e.g., "Here's an enhanced prompt:\n\nA cute cat..." -> "A cute cat..."
        """
        cleaned_text = text.strip()

        # Split the text into lines to analyze the first line
        lines = cleaned_text.splitlines()
        if not lines:
            return ""

        first_line = lines[0].strip()

        # Keywords that often appear in boilerplate preambles
        boilerplate_keywords = ["here's", "here is", "enhanced", "revised", "expanded", "prompt", "possible"]

        # Heuristic: Check if the first line looks like a boilerplate preamble.
        # Condition: It ends with a colon and contains at least one of the keywords.
        # This is safer than just checking for a colon, as a valid prompt could contain one.
        if first_line.endswith(':') and any(keyword in first_line.lower() for keyword in boilerplate_keywords):
            # If it matches, we assume the actual content starts after this line.
            # Join the rest of the lines and strip any leading whitespace/newlines.
            cleaned_text = '\n'.join(lines[1:]).lstrip()
        
        return cleaned_text
    

    def brekel_enhance_prompt(self, prompt: str, prefix: str, model_name: str, quantization: str, memory_management: str, system_prompt: str, target_length: int, creativity: float, seed: int, postfix: str, unique_id=None):
        """
        The main execution function of the node. It orchestrates loading the model,
        generating the enhanced prompt, and handling memory management.
        """
        global _loaded_model, _loaded_tokenizer, _loaded_model_name, _model_is_offloaded, _loaded_quantization
        
        try:
            # --- INPUT VALIDATION ---
            if model_name in ["NO_MODEL_FOUND", "ERROR_SCANNING_MODELS"]:
                raise ValueError("No valid LLM models found in ComfyUI/models/LLM. Please check your setup.")
            
            all_system_prompts = get_system_prompts()
            active_system_prompt = all_system_prompts.get(system_prompt)
            if not active_system_prompt or "ERROR" in active_system_prompt:
                raise ValueError(f"Selected system prompt '{system_prompt}' could not be loaded. Check for errors above.")

            # --- MODEL LOADING ---
            self._load_model(model_name, quantization)
            
            if _loaded_model is None or _loaded_tokenizer is None:
                 raise RuntimeError("Model and/or tokenizer failed to load, but no exception was caught. Check logs.")

            if seed != 0: torch.manual_seed(seed)

            # --- PROMPT GENERATION ---
            logger.info(f"\n--- Enhancing Prompt ---")
            logger.info(f"Original: '{prompt}'")
            logger.info(f"Settings: model={model_name}, creativity={creativity}, target_length={target_length}, seed={'random' if seed == 0 else seed}")
            
            length_instruction = (
                f"\n\nIMPORTANT: Aim for a total length of approximately {target_length} characters. Your response must be ONLY the enhanced prompt itself, without any conversational introduction."
            )
            final_system_prompt = active_system_prompt + length_instruction
            logger.info(f"\nSystem prompt: '{final_system_prompt}'\n")

            messages = [{"role": "system", "content": final_system_prompt}, {"role": "user", "content": f"user_prompt: {prompt}"}]
            text_input_for_llm = _loaded_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = _loaded_tokenizer(text_input_for_llm, return_tensors="pt").to(_loaded_model.device)
            
            # Set generation parameters. We only need `max_new_tokens` to control the output length.
            generation_params = {"max_new_tokens": target_length, "pad_token_id": _loaded_tokenizer.eos_token_id}

            if creativity > 0.0:
                generation_params.update({"do_sample": True, "temperature": creativity, "top_p": 0.9})
            else:
                generation_params["do_sample"] = False

            with torch.inference_mode():
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "The following generation flags are not valid.*")
                    outputs = _loaded_model.generate(**model_inputs, **generation_params)
                
                input_token_length = model_inputs.input_ids.shape[-1]
                generated_ids = outputs[0][input_token_length:]
                raw_enhanced_prompt = _loaded_tokenizer.decode(generated_ids, skip_special_tokens=True)

            enhanced_prompt = self._clean_llm_output(raw_enhanced_prompt)
            enhanced_prompt = enhanced_prompt.strip().strip('"').rstrip(' ,.').strip()
            
            prompt_parts = [p.strip() for p in [prefix, enhanced_prompt, postfix] if p.strip()]
            final_prompt = ", ".join(prompt_parts)

            # Send the prompt content to the UI
            if unique_id and PromptServer.instance:
                text_to_display = f"""<div style="margin-bottom: 4px; font-size: 0.8em; color: #888;">Prompt: {final_prompt}</div>"""
                PromptServer.instance.send_progress_text(final_prompt, unique_id)

            logger.info(f"\n--- Final Combined Prompt ---\n{final_prompt}\n\n")
            return (final_prompt,)

        except Exception as e:
            logger.error(f"!!! PROMPT ENHANCEMENT FAILED: {e}")
            raise

        finally:
            if _loaded_model is not None:
                if memory_management == "Offload to CPU after use":
                    if not _model_is_offloaded:
                        if quantization == "Disabled" and _loaded_model.device.type == 'cuda':
                            logger.info(f"Offloading model '{_loaded_model_name}' to CPU RAM...")
                            start_time = time.perf_counter()
                            _loaded_model.to("cpu")
                            logger.info(f"Model offloaded to CPU in {time.perf_counter() - start_time:.2f} seconds.")
                            _model_is_offloaded = True
                        else:
                            logger.info(f"Model '{_loaded_model_name}' is quantized or already on CPU. Marking as 'offloaded'.")
                            _model_is_offloaded = True
                
                elif memory_management == "Unload completely after use":
                    logger.info(f"Unloading model '{_loaded_model_name}' from all memory...")
                    del _loaded_model, _loaded_tokenizer
                    _loaded_model = _loaded_tokenizer = _loaded_model_name = _loaded_quantization = None
                    _model_is_offloaded = False
                    torch.cuda.empty_cache()


# --- ComfyUI Node Registration ---
NODE_CLASS_MAPPINGS = {"BrekelEnhancePrompt": BrekelEnhancePrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"BrekelEnhancePrompt": "Brekel Prompt Enhancer (LLM)"}