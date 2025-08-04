#
# Brekel Prompt Chooser Node for ComfyUI
# Version: 1.0.0
#
# Author: Brekel - https://brekel.com
#
# This custom node chooses a prompt text file stored from a folder
#
# Key Features:
# - Choose a random prompt from a folder using a seed.
# - Choose a specific prompt from a folder by its index.


import os
import logging
import random


# --- CONFIGURATION CONSTANT ---
# Define the subfolder name where text files are stored.
SUBFOLDER_NAME = "prompt_chooser"


# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Default Folder Path ---
SCRIPT_DIR = os.path.dirname(__file__)
DEFAULT_PROMPTS_DIR = os.path.join(SCRIPT_DIR, SUBFOLDER_NAME)


# --- PROMPT CHOOSER CLASS ---
class BrekelPromptChooser:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": DEFAULT_PROMPTS_DIR,
                    "tooltip": "Path to the folder containing your .txt prompt files."
                }),
                "selection_mode": (["Random", "Index"], {
                    "default": "Random",
                    "tooltip": "Choose 'Random' to select a file randomly based on the seed. Choose 'Index' to select a specific file."
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFF}),
                "file_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4095, # A reasonably high number for the index
                    "tooltip": "The index of the file to choose when in 'Index' mode (sorted alphabetically, if index is higher than the number of files it wraps back around)."
                }),
            }
        }

    # --- Node configuration for ComfyUI ---
    CATEGORY = "Brekel"
    FUNCTION = "choose_prompt"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)

    def choose_prompt(self, folder_path: str, selection_mode: str, seed: int, file_index: int):
        """
        Main execution function. It selects a .txt file from the given folder,
        either randomly or by index, and returns its content.
        """
        # Check if the directory exists. It should be part of the repo.
        if not os.path.isdir(folder_path):
            error_msg = f"Prompt folder not found at '{folder_path}'. Please check the path or reinstall the node."
            logger.error(f"[Brekel Prompt Chooser] {error_msg}")
            return (f"ERROR: {error_msg}",)

        try:
            available_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])
        except Exception as e:
            error_msg = f"Could not scan folder '{folder_path}': {e}"
            logger.error(f"[Brekel Prompt Chooser] {error_msg}")
            return (f"ERROR: {error_msg}",)
        
        if not available_files:
            error_msg = f"No .txt files found in '{folder_path}'"
            logger.error(f"[Brekel Prompt Chooser] {error_msg}")
            return (f"ERROR: {error_msg}",)

        chosen_file = None
        
        # Logic to handle the different selection modes.
        if selection_mode == "Random":
            # Seed the random number generator for deterministic results.
            random.seed(seed)
            # Choose a random file from the list.
            chosen_file = random.choice(available_files)
            print(f"[Brekel Prompt Chooser] Random mode (seed {seed}) selected '{chosen_file}' from '{folder_path}'")

        elif selection_mode == "Index":
            num_files = len(available_files)
            # Use modulo to wrap the index around if it's too high.
            actual_index = file_index % num_files
            chosen_file = available_files[actual_index]
            print(f"[Brekel Prompt Chooser] Index mode (input index {file_index} -> wrapped to {actual_index}) selected '{chosen_file}' from '{folder_path}'")
        
        else:
            # Fallback in case of an unexpected mode.
            error_msg = f"Unknown selection mode: {selection_mode}"
            logger.error(f"[Brekel Prompt Chooser] {error_msg}")
            return (f"ERROR: {error_msg}",)

        # Read the content of the chosen file.
        file_path = os.path.join(folder_path, chosen_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            return (content,)
        except Exception as e:
            error_msg = f"Failed to read file '{file_path}': {e}"
            logger.error(f"[Brekel Prompt Chooser] {error_msg}")
            return (f"ERROR: {error_msg}",)


# --- ComfyUI Node Registration ---
NODE_CLASS_MAPPINGS = {"BrekelPromptChooser": BrekelPromptChooser,}
NODE_DISPLAY_NAME_MAPPINGS = {"BrekelPromptChooser": "Brekel Prompt Chooser",}