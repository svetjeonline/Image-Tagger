# -*- coding: utf-8 -*-
# Version 7: Desktop App Focus - Structure of panels, Menu, EN AI Output (200 chars, more keywords target)
import tkinter as tk
from tkinter import Menu  # For standard menu bar
import tkinter.filedialog as fd
import tkinter.messagebox as messagebox
import customtkinter as ctk
import os
import platform
import json
import logging
import traceback
import threading
from queue import Queue, Empty
from PIL import Image, ImageTk, ExifTags
from typing import Optional, Dict, Any, List, Tuple, Callable
import time
import contextlib  # For torch.no_grad()

# --- App Version ---
APP_VERSION = "7.0"

# --- Dependency Checks & Setup ---
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    print("*" * 50)
    print("WARNING: The PyTorch library is not installed.")
    print("GPU acceleration will not be available. AI will run on CPU.")
    print("Installation: Follow the instructions at https://pytorch.org/")
    print("*" * 50)

try:
    from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
    transformers_available = True
except ImportError:
    transformers_available = False
    print("*" * 50)
    print("CRITICAL ERROR: The Transformers library is not installed.")
    print("The application cannot function without it.")
    print("Installation: pip install transformers")
    print("*" * 50)
    # exit()  # Consider exiting if essential

try:
    from iptcinfo3 import IPTCInfo, IPTCError, IsFileValidIPTCIptcinfoError
    iptc_available = True
except ImportError:
    iptc_available = False
    IPTCInfo, IPTCError, IsFileValidIPTCIptcinfoError = None, Exception, Exception
    print("INFO: The iptcinfo3 library is not available, IPTC metadata will not be processed.")

# --- XMP Import with Exempi Check ---
try:
    # This import sequence can raise ExempiLoadError if libexempi is missing
    from libxmp import XMPFiles, consts as XMPConsts, XMPError, XMPMeta
    from libxmp.utils import file_to_dict  # For easier reading
    # Check if it *really* loaded - sometimes import succeeds but core fails later
    if not XMPMeta:
        raise ImportError("XMPMeta not loaded correctly.")
    xmp_available = True
    print("INFO: The python-xmp-toolkit library (and Exempi) successfully loaded.")
except ImportError:
    # This catches if python-xmp-toolkit itself is not installed
    print("*" * 50)
    print("WARNING: The python-xmp-toolkit library is not installed.")
    print("XMP metadata will not be read or written.")
    print("Installation: pip install python-xmp-toolkit")
    print("*" * 50)
    xmp_available = False
    XMPFiles, XMPConsts, XMPError, XMPMeta = None, None, Exception, None
except Exception as e:
    # This catches ExempiLoadError and other potential issues during import
    print("*" * 50)
    print(f"WARNING: Error importing XMP library: {e}")
    print("XMP metadata will not be read or written.")
    if "Exempi library not found" in str(e):
        print("It seems that the system library 'Exempi' is missing.")
        print("Installation of the system library 'exempi':")
        print("  Linux (Debian/Ubuntu): sudo apt-get install libexempi8")  # Updated package name might be needed
        print("  Linux (Fedora): sudo dnf install exempi")
        print("  macOS (Homebrew): brew install exempi")
        print("  Windows: Requires manual installation of Exempi (e.g., from vcpkg, msys2, or finding a binary).")
    else:
        print("Ensure that the python-xmp-toolkit library is correctly installed and functional.")
    print("*" * 50)
    xmp_available = False
    XMPFiles, XMPConsts, XMPError, XMPMeta = None, None, Exception, None

# --- Logging Configuration ---
log_formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
LOG_FILENAME = "image_tagger_app_v7.log"
# Use 'a' mode to append logs across runs, 'w' to overwrite each time
log_handler_file = logging.FileHandler(LOG_FILENAME, encoding='utf-8', mode='a')
log_handler_file.setFormatter(log_formatter)
log_handler_stream = logging.StreamHandler()
log_handler_stream.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set to DEBUG for more verbose logs
logger.addHandler(log_handler_file)
logger.addHandler(log_handler_stream)

# --- Global Variables and Constants ---
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.tiff', '.tif')
THUMBNAIL_SIZE = (80, 80)
PREVIEW_SIZE = (400, 400)
CONFIG_FILE = "tagger_config_v7.json"
DEFAULT_STOP_WORDS_EN = set([  # English stopwords for processing English captions
    "a", "an", "the", "in", "on", "at", "to", "of", "for", "with", "this", "that", "is", "are", "was", "were",
    "it", "he", "she", "they", "image", "picture", "photo", "photograph", "illustration", "drawing", "background",
    "foreground", "close", "up", "view", "of", "and", "with", "a", "on", "in", "it's", "its", "containing", "shows",
    "there", "two", "one", "group", "several", "man", "woman", "people", "person", "closeup", "shot", "landscape",
    "portrait", "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
    "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
    "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves", "depicting", "featuring", "displaying", "showcasing", "including", "standing", "sitting", "looking",
    "white", "black", "red", "green", "blue", "yellow", "orange", "purple", "brown", "gray", "large", "small", "big", "tall",
    "short", "long", "wide", "narrow", "high", "low", "top", "bottom", "left", "right", "center", "middle", "front", "back"
])

# Status Constants
STATUS_PENDING = "Pending"
STATUS_LOADING_META = "Loading Metadata"
STATUS_LOADING_THUMB = "Loading Thumbnail"
STATUS_READY = "Ready"
STATUS_QUEUED_AI = "[AI] Queued"
STATUS_PROCESSING_AI = "[AI] Processing"
STATUS_PROCESSED_OK = "[AI] Completed"
STATUS_PROCESSED_ERR = "[AI] Error"
STATUS_SAVING = "[IO] Saving"
STATUS_SAVED = "[OK] Saved"
STATUS_SAVE_ERROR = "[ERR] Save Error"
STATUS_LOAD_ERROR = "[ERR] Load Error"
STATUS_THUMB_ERROR = "[ERR] Thumbnail Error"

STATUS_COLORS = {
    STATUS_PENDING: "gray", STATUS_LOADING_META: "gray", STATUS_LOADING_THUMB: "gray",
    STATUS_READY: "white", STATUS_QUEUED_AI: "orange", STATUS_PROCESSING_AI: "lightblue",
    STATUS_PROCESSED_OK: "lightgreen", STATUS_PROCESSED_ERR: "red", STATUS_SAVING: "lightblue",
    STATUS_SAVED: "green", STATUS_SAVE_ERROR: "red", STATUS_LOAD_ERROR: "red",
    STATUS_THUMB_ERROR: "red",
}

# --- AI Models ---
AVAILABLE_MODELS = {
    "BLIP Base (EN output)": "Salesforce/blip-image-captioning-base",
    "BLIP Large (EN output)": "Salesforce/blip-image-captioning-large",
}
loaded_ai_models: Dict[str, Dict[str, Any]] = {}
ai_initialization_lock = threading.Lock()

# --- Utility Functions ---
def safe_decode(byte_string: Optional[bytes], encoding: str = 'utf-8') -> Optional[str]:
    if byte_string is None:
        return None
    try:
        return byte_string.decode(encoding, 'replace')
    except Exception:
        return None

def get_device(requested_device: str) -> str:
    if not torch_available:
        if requested_device != "cpu":
            logger.warning("PyTorch is not available, forcing CPU.")
        return "cpu"
    if requested_device == 'auto':
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    elif requested_device == 'cuda':
        if torch.cuda.is_available():
            return "cuda:0"
        else:
            logger.warning("CUDA requested but not available, falling back to CPU.")
            return "cpu"
    else:
        return "cpu"

# --- Data Class ---
class ImageFile:
    """Stores data and status for a single image."""
    def __init__(self, file_path: str):
        self.path: str = file_path
        self.base_name: str = os.path.basename(file_path)
        self.status: str = STATUS_PENDING
        self.status_details: str = ""
        self.ctk_thumbnail: Optional[ctk.CTkImage] = None
        self.list_item_widget: Optional['ImageListItem'] = None
        self.existing_meta: Dict[str, Any] = {"caption": None, "keywords": [], "source": None}
        self.generated_meta: Dict[str, Any] = {"caption": None, "keywords": []}  # AI output (expected English)
        self.edited_meta: Optional[Dict[str, Any]] = None  # User edits
        self.is_dirty: bool = False

    def set_list_item_widget(self, widget: 'ImageListItem'):
        self.list_item_widget = widget

    def update_status(self, status: str, details: str = ""):
        self.status = status
        self.status_details = details
        logger.debug(f"Status update for {self.base_name}: {status} ({details})")
        if self.list_item_widget and self.list_item_widget.winfo_exists():
            # Use app's safe_widget_update via the stored app reference
            self.list_item_widget.app.after(0, self.list_item_widget.update_display)

    def set_existing_meta(self, caption: Optional[str], keywords: List[str], source: Optional[str]):
        self.existing_meta = {"caption": caption, "keywords": keywords, "source": source}
        if self.status == STATUS_LOADING_META:
            self.update_status(STATUS_READY)

    def set_generated(self, caption: Optional[str], keywords: Optional[List[str]]):
        """Sets AI generated data (expected in English)."""
        self.generated_meta["caption"] = caption.strip() if caption else ""
        self.generated_meta["keywords"] = sorted([kw.strip().lower() for kw in keywords if kw.strip()]) if keywords else []
        self.edited_meta = None
        self.is_dirty = True
        status = STATUS_PROCESSED_OK if (caption or keywords) else STATUS_PROCESSED_ERR
        details = "" if status == STATUS_PROCESSED_OK else "AI returned no data"
        self.update_status(status, details=details)

    def update_edited(self, caption: str, keywords: List[str]):
        """Updates the edited metadata from user input."""
        new_caption = caption.strip()
        new_keywords = sorted([kw.strip() for kw in keywords if kw.strip()])
        current_to_compare = self.edited_meta if self.edited_meta is not None else self.generated_meta
        if new_caption != current_to_compare.get("caption", "") or new_keywords != current_to_compare.get("keywords", []):
            self.edited_meta = {"caption": new_caption, "keywords": new_keywords}
            if not self.is_dirty:
                self.is_dirty = True
                logger.debug(f"Marked {self.base_name} as dirty.")
                # Update detail panel dirty indicator via app if this is the selected file
                if self.list_item_widget and self.list_item_widget.app.currently_selected_file == self.path:
                    self.list_item_widget.app.after(0, self.list_item_widget.app.update_detail_panel_dirty_indicator)
                # Update main action buttons via app
                self.list_item_widget.app.after(0, self.list_item_widget.app.update_action_buttons_state)

    def get_display_caption(self) -> str:
        if self.edited_meta is not None:
            return self.edited_meta.get("caption", "")
        if self.generated_meta.get("caption") is not None:
            return self.generated_meta.get("caption", "")
        return self.existing_meta.get("caption", "") or ""

    def get_display_keywords(self) -> List[str]:
        if self.edited_meta is not None:
            return self.edited_meta.get("keywords", [])
        if self.generated_meta.get("keywords"):
            return self.generated_meta.get("keywords", [])
        return self.existing_meta.get("keywords", []) or []

    def get_data_to_save(self) -> Dict[str, Any]:
        data_source = self.edited_meta if self.edited_meta is not None else self.generated_meta
        return {
            "caption": data_source.get("caption", ""),
            "keywords": [str(kw) for kw in data_source.get("keywords", [])]  # Ensure strings
        }

    def mark_saved(self):
        self.is_dirty = False
        # Keep edited_meta until new generation or load? Or clear it?
        # Let's clear it to reflect the saved state matches generated/edited
        # self.edited_meta = None  # Or maybe keep it to show what was last saved? Let's keep it for now.
        self.update_status(STATUS_SAVED)
        if self.list_item_widget and self.list_item_widget.app.currently_selected_file == self.path:
            self.list_item_widget.app.after(0, self.list_item_widget.app.display_details(self))

    def mark_save_error(self, error_msg: str):
        self.update_status(STATUS_SAVE_ERROR, details=error_msg)
        self.is_dirty = True  # Still dirty as save failed

# --- AI Model Handling ---
def initialize_ai_model(model_name: str, device: str) -> Optional[Dict[str, Any]]:
    if not transformers_available:
        log_message("Transformers library is not available.", level=logging.CRITICAL)
        return None
    if model_name not in AVAILABLE_MODELS:
        log_message(f"Unknown model name: {model_name}", level=logging.ERROR)
        return None
    model_id = AVAILABLE_MODELS[model_name]
    cache_key = f"{model_id}_{device}"
    if cache_key in loaded_ai_models:
        return loaded_ai_models[cache_key]
    with ai_initialization_lock:
        if cache_key in loaded_ai_models:
            return loaded_ai_models[cache_key]
        try:
            log_message(f"Initializing AI model: {model_name} ({model_id}) on {device}...", level=logging.INFO)
            processor = BlipProcessor.from_pretrained(model_id)
            model = BlipForConditionalGeneration.from_pretrained(model_id)
            log_message(f"Moving model {model_name} to device {device}...", level=logging.INFO)
            model.to(device)
            model.eval()
            loaded_ai_models[cache_key] = {"processor": processor, "model": model, "device": device}
            log_message(f"AI model {model_name} successfully initialized on {device}.", level=logging.INFO)
            return loaded_ai_models[cache_key]
        except Exception as e:
            log_message(f"CRITICAL ERROR: Failed to initialize AI model {model_name} ({model_id}) on {device}: {e}", level=logging.CRITICAL, exc_info=True)
            # Use global app instance if available to show messagebox
            global app
            if 'app' in globals() and app is not None and app.winfo_exists():
                app.after(0, lambda: messagebox.showerror("AI Initialization Error", f"Failed to load AI model '{model_name}'.\nCheck the log ({LOG_FILENAME}).\n\nError: {e}", parent=app))
            return None

def generate_caption_and_keywords(image_path: str, model_components: Dict[str, Any], stop_words: set) -> Tuple[Optional[str], Optional[List[str]]]:
    """Generates ENGLISH caption (up to 200 chars) and keywords."""
    if not model_components:
        log_message("ERROR: AI model components are not available for generation.", level=logging.ERROR)
        return None, None
    processor, model, device = model_components["processor"], model_components["model"], model_components["device"]
    base_name = os.path.basename(image_path)
    try:
        logger.debug(f"Processing image {base_name} on {device} for EN output (max 200)...")
        img = Image.open(image_path).convert('RGB')
        inputs = processor(images=img, return_tensors="pt").to(device)
        gen_kwargs = {"max_length": 200, "num_beams": 5, "early_stopping": True}
        # Use torch.no_grad() for inference if torch is available
        context_manager = torch.no_grad() if torch_available else contextlib.nullcontext()
        with context_manager:
            out = model.generate(**inputs, **gen_kwargs)
        caption = processor.decode(out[0], skip_special_tokens=True).strip()
        logger.debug(f"Raw EN caption for {base_name} ({len(caption)} chars): {caption}")
        # Keyword Extraction
        words = [word.lower().strip('.,!?"\'()[]{}') for word in caption.split() if word.lower() not in stop_words and len(word) > 2]
        keywords = sorted(list(set(words)))
        logger.info(f"Generated EN description ({len(caption)} chars) and {len(keywords)} keywords for {base_name}.")
        return caption, keywords
    except Exception as e:
        log_message(f"ERROR generating EN description/keywords for {base_name}: {e}", level=logging.ERROR, exc_info=True)
        return None, None

# --- Metadata Handling ---
def read_metadata(file_path: str) -> Tuple[Optional[str], List[str], Optional[str]]:
    """
    Reads IPTC and XMP metadata, tries to return primary set of description/keywords.
    Returns: (caption, keywords, source)
    Source indicates preference ('IPTC', 'XMP', 'Both', None)
    """
    iptc_data = {"caption": None, "keywords": []}
    xmp_data = {"caption": None, "keywords": []}
    base_name = os.path.basename(file_path)

    # --- Reading IPTC ---
    if iptc_available and IPTCInfo:
        try:
            logger.debug(f"Reading IPTC for {base_name}")
            info = IPTCInfo(file_path, force=False, inp_charset='utf8')  # Assume UTF-8 input
            caption_bytes = info.get('caption/abstract')
            iptc_data["caption"] = safe_decode(caption_bytes) if caption_bytes else None

            keywords_bytes = info.get('keywords', [])
            iptc_data["keywords"] = [kw for kw in (safe_decode(k) for k in keywords_bytes) if kw]
        except (IPTCError, IsFileValidIPTCIptcinfoError, FileNotFoundError) as e:
            logger.debug(f"IPTC read issue (normal) for {base_name}: {e}")
            pass  # Ignore common read errors
        except Exception as e:
            logger.warning(f"Error reading IPTC from {base_name}: {e}", exc_info=True)

    # --- Reading XMP ---
    if xmp_available and XMPFiles and XMPMeta:  # Check XMPMeta too
        xmpfile = None
        try:
            logger.debug(f"Reading XMP for {base_name}")
            xmpfile = XMPFiles(file_path=file_path, open_forupdate=False)
            xmp = xmpfile.get_xmp()
            if xmp:
                # Description (prefer dc:description, fallback photoshop:Headline)
                caption_val = None
                try:
                    if xmp.does_property_exist(XMPConsts.XMP_NS_DC, 'description'):
                        loc_text = xmp.get_localized_text(XMPConsts.XMP_NS_DC, 'description', None, 'x-default')
                        if loc_text and loc_text[1]:
                            caption_val = loc_text[1]
                        else:  # Fallback if x-default is empty or not present
                            props = xmp.get_property(XMPConsts.XMP_NS_DC, 'description')
                            if props:
                                caption_val = str(props)  # Might be complex, just stringify
                except XMPError as desc_err:
                    logger.debug(f"XMP description read issue: {desc_err}")
                except Exception as desc_err_unex:
                    logger.warning(f"Unexpected XMP desc read error: {desc_err_unex}")

                if not caption_val:
                    try:
                        if xmp.does_property_exist(XMPConsts.XMP_NS_PHOTOSHOP, 'Headline'):
                            caption_val = xmp.get_property(XMPConsts.XMP_NS_PHOTOSHOP, 'Headline')
                    except XMPError as headline_err:
                        logger.debug(f"XMP headline read issue: {headline_err}")
                    except Exception as headline_err_unex:
                        logger.warning(f"Unexpected XMP headline read error: {headline_err_unex}")

                xmp_data["caption"] = caption_val.strip() if caption_val else None

                # Keywords (dc:subject array)
                try:
                    if xmp.does_property_exist(XMPConsts.XMP_NS_DC, 'subject'):
                        keywords_list = []
                        # Iterate through the array items
                        count = xmp.count_array_items(XMPConsts.XMP_NS_DC, 'subject')
                        for i in range(1, count + 1):
                            item = xmp.get_array_item(XMPConsts.XMP_NS_DC, 'subject', i)
                            if item:
                                keywords_list.append(item)
                        xmp_data["keywords"] = [kw.strip() for kw in keywords_list if kw.strip()]
                except XMPError as subj_err:
                    logger.debug(f"XMP subject read issue: {subj_err}")
                except Exception as subj_err_unex:
                    logger.warning(f"Unexpected XMP subject read error: {subj_err_unex}")

        except XMPError as e:
            # Filter out common "not found" or parse errors which are expected
            if "does not contain XMP data" not in str(e) and "Failed to parse" not in str(e):
                logger.warning(f"Error reading XMP from {base_name}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error reading XMP from {base_name}: {e}", exc_info=True)
        finally:
            if xmpfile:
                xmpfile.close_file()

    # --- Decision on primary source ---
    has_xmp = bool(xmp_data["caption"] or xmp_data["keywords"])
    has_iptc = bool(iptc_data["caption"] or iptc_data["keywords"])

    # Prefer XMP if it exists, otherwise IPTC
    if has_xmp:
        logger.debug(f"Using XMP as primary source for {base_name}")
        return xmp_data["caption"], xmp_data["keywords"], "XMP" if not has_iptc else "Both"
    elif has_iptc:
        logger.debug(f"Using IPTC as primary source for {base_name}")
        return iptc_data["caption"], iptc_data["keywords"], "IPTC"
    else:
        logger.debug(f"No existing metadata found for {base_name}")
        return None, [], None

def write_metadata(file_path: str, data_to_write: Dict[str, Any], settings: Dict[str, Any]) -> bool:
    """Writes metadata (IPTC and/or XMP) based on settings. Returns True on success."""
    success = True
    write_iptc = settings.get("write_iptc", False) and iptc_available
    write_xmp = settings.get("write_xmp", False) and xmp_available
    append_keywords = settings.get("append_keywords", False)
    base_name = os.path.basename(file_path)

    caption = data_to_write.get("caption", "")
    keywords_raw = data_to_write.get("keywords", [])
    keywords = [str(kw).strip() for kw in keywords_raw if str(kw).strip()]

    if not caption and not keywords:
        logger.info(f"No data (description/keywords) to write for {base_name}.")
        # Return True because there was nothing to *fail* at writing
        return True

    final_keywords = keywords
    if append_keywords and (write_iptc or write_xmp):
        try:
            # Read existing metadata specifically for appending
            _, existing_keywords, _ = read_metadata(file_path)
            if existing_keywords:
                # Case-insensitive check for uniqueness when appending
                existing_set_lower = set(k.lower() for k in existing_keywords)
                combined_keywords = list(existing_keywords)  # Start with existing
                for kw in keywords:
                    if kw.lower() not in existing_set_lower:
                        combined_keywords.append(kw)
                final_keywords = combined_keywords
                logger.debug(f"Appending keywords for {base_name}. Original: {len(existing_keywords)}, New: {len(keywords)}, Final: {len(final_keywords)}")
            else:
                # No existing keywords, just use the new ones
                final_keywords = keywords
        except Exception as e:
            logger.warning(f"Failed to read existing keywords for appending in {base_name}: {e}. Overwriting.")
            final_keywords = keywords  # Fallback to overwriting if read fails

    # --- Writing IPTC ---
    if write_iptc and IPTCInfo:
        logger.debug(f"Writing IPTC for {base_name}...")
        iptc_success = False
        try:
            info = IPTCInfo(file_path, force=True, inp_charset='utf8', out_charset='utf8')
            # Handle caption: write if provided, delete if empty and exists
            if caption:
                info['caption/abstract'] = caption.encode('utf-8', 'replace')
            elif 'caption/abstract' in info:
                del info['caption/abstract']
            # Handle keywords: write if provided, delete if empty and exists
            if final_keywords:
                info['keywords'] = [kw.encode('utf-8', 'replace') for kw in final_keywords]
            elif 'keywords' in info:
                del info['keywords']
            # Save changes
            info.save()
            logger.debug(f"IPTC write successful for {base_name}")
            iptc_success = True
        except Exception as e:
            log_message(f"ERROR writing IPTC to {base_name}: {e}", level=logging.ERROR, exc_info=True)
        finally:
            if not iptc_success:
                success = False  # Mark overall failure if IPTC fails

    # --- Writing XMP ---
    if write_xmp and XMPFiles and XMPMeta:
        logger.debug(f"Writing XMP for {base_name}...")
        xmp_success = False
        xmpfile = None
        try:
            xmpfile = XMPFiles(file_path=file_path, open_forupdate=True)
            xmp = xmpfile.get_xmp()
            if not xmp:
                logger.debug(f"Creating new XMP metadata for {base_name}")
                xmp = XMPMeta()
                # Register namespaces needed if creating new XMP block
                try:
                    xmp.register_namespace(XMPConsts.XMP_NS_DC, "dc")
                    xmp.register_namespace(XMPConsts.XMP_NS_PHOTOSHOP, "photoshop")  # If using headline etc.
                    # Add other namespaces if needed (e.g., XMP_NS_IPTCCORE)
                except XMPError as reg_err:
                    log_message(f"Error registering XMP ns: {reg_err}", logging.ERROR)
                    raise  # Propagate error if registration fails

            temp_success = True  # Track errors within XMP property setting
            # Description (dc:description, x-default)
            try:
                # Delete existing description property entirely (handles language alternatives)
                if xmp.does_property_exist(XMPConsts.XMP_NS_DC, 'description'):
                    xmp.delete_property(XMPConsts.XMP_NS_DC, 'description')
                # Set new value if caption is not empty
                if caption:
                    xmp.set_localized_text(XMPConsts.XMP_NS_DC, 'description', None, 'x-default', caption)
            except XMPError as desc_err:
                log_message(f"Error writing XMP description: {desc_err}", logging.ERROR)
                temp_success = False

            # Keywords (dc:subject array)
            try:
                # Delete existing subject property entirely
                if xmp.does_property_exist(XMPConsts.XMP_NS_DC, 'subject'):
                    xmp.delete_property(XMPConsts.XMP_NS_DC, 'subject')
                # Append new keywords if list is not empty
                if final_keywords:
                    # Ensure it's created as an unordered array
                    xmp.create_array(XMPConsts.XMP_NS_DC, 'subject', XMPConsts.XMP_PROP_ARRAY_IS_UNORDERED)
                    for kw in final_keywords:
                        xmp.append_array_item(XMPConsts.XMP_NS_DC, 'subject', str(kw))  # No extra options needed here
            except XMPError as kw_err:
                log_message(f"Error writing XMP keywords: {kw_err}", logging.ERROR)
                temp_success = False

            # Only attempt to save if setting properties didn't raise errors
            if temp_success:
                if xmpfile.can_put_xmp(xmp):
                    xmpfile.put_xmp(xmp)
                    logger.debug(f"XMP write successful for {base_name}")
                    xmp_success = True
                else:
                    log_message(f"ERROR: Cannot write XMP to {base_name} (format not supported?).", logging.ERROR)
            else:
                logger.error(f"Skipping final XMP write for {base_name} due to errors in setting properties.")

        except Exception as e:
            log_message(f"ERROR writing XMP to {base_name}: {e}", level=logging.ERROR, exc_info=True)
        finally:
            if xmpfile:
                xmpfile.close_file()
            if not xmp_success:
                success = False  # Mark overall failure if XMP fails

    # Log final outcome
    if success:
        formats = []
        if write_iptc:
            formats.append("IPTC")
        if write_xmp:
            formats.append("XMP")
        if formats:
            logger.info(f"Metadata written ({', '.join(formats)}) for {base_name}")
        # else: logger.debug(f"No metadata formats enabled for writing to {base_name}")  # Already logged if no data
    else:
        logger.error(f"Failed to write metadata for {base_name}")

    return success

# --- Worker Threads ---
def load_files_and_metadata_thread(folder_path: str, progress_queue: Queue, stop_event: threading.Event):
    """Finds files, creates ImageFile objects, and loads existing metadata."""
    log_message(f"Starting folder scan and metadata loading: {folder_path}")
    found_files_paths = []
    try:
        for root, _, files in os.walk(folder_path):
            if stop_event.is_set():
                log_message("Folder scan interrupted.", logging.WARNING)
                progress_queue.put({"type": "file_scan_stopped"})
                return
            for file in files:
                if file.lower().endswith(SUPPORTED_EXTENSIONS):
                    full_path = os.path.join(root, file)
                    found_files_paths.append(full_path)
    except Exception as e:
        log_message(f"Error traversing folder {folder_path}: {e}", level=logging.ERROR, exc_info=True)
        progress_queue.put({"type": "error", "data": {"message": f"Folder traversal error: {e}", "thread_name": "FileMetaLoader"}})
        return

    total_files = len(found_files_paths)
    log_message(f"Found {total_files} supported images.")
    progress_queue.put({"type": "file_scan_found_count", "data": {"count": total_files}})

    processed_count = 0
    files_data = []  # Collect data before sending to queue
    for i, file_path in enumerate(found_files_paths):
        if stop_event.is_set():
            log_message("Metadata loading interrupted.", logging.WARNING)
            progress_queue.put({"type": "metadata_load_stopped"})
            return
        try:
            # Send pending message first for immediate UI feedback
            progress_queue.put({"type": "file_found_pending", "data": {"path": file_path, "base_name": os.path.basename(file_path)}})
            # Now read metadata
            caption, keywords, source = read_metadata(file_path)
            files_data.append({
                "path": file_path, "base_name": os.path.basename(file_path),
                "existing_caption": caption, "existing_keywords": keywords, "existing_source": source,
            })
            processed_count += 1
            # Update progress periodically
            if (i + 1) % 20 == 0 or (i + 1) == total_files:
                progress_queue.put({"type": "metadata_load_progress", "data": {"progress": (i + 1) / total_files}})
        except Exception as e:
            log_message(f"Error processing file {file_path} during metadata loading: {e}", level=logging.ERROR, exc_info=True)
            # Send status update for the specific file that failed
            progress_queue.put({"type": "update_status_list", "data": {"path": file_path, "status": STATUS_LOAD_ERROR, "details": str(e)}})

    log_message(f"Metadata loading completed for {processed_count}/{total_files} files.")
    # Send all collected data at once
    progress_queue.put({"type": "metadata_load_done", "data": {"files_data": files_data, "count": processed_count}})

def load_thumbnails_thread(image_files: List[ImageFile], progress_queue: Queue):
    """Loads thumbnails for given ImageFile objects."""
    log_message(f"Starting background thumbnail loading for {len(image_files)} images...")
    loaded_count = 0
    total_thumbs = len(image_files)
    for image_file in image_files:
        file_path = image_file.path
        try:
            # Update status before starting load for this thumb
            progress_queue.put({"type": "update_status_list", "data": {"path": file_path, "status": STATUS_LOADING_THUMB}})
            img = Image.open(file_path)
            orientation = 1  # Default orientation
            try:
                # Safely get exif data and orientation tag
                exif_data = img.getexif()
                if exif_data:
                    orientation = exif_data.get(0x0112, 1)  # 0x0112 is the EXIF Orientation tag
            except Exception as exif_err:
                # Log exif errors at debug level as they are common
                logger.debug(f"EXIF read error for {image_file.base_name}: {exif_err}")

            # Apply rotation based on orientation tag
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)

            # Create thumbnail
            img.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)

            # Ensure image is in RGB or RGBA for CTkImage
            if img.mode not in ('RGB', 'RGBA'):
                logger.debug(f"Converting image {image_file.base_name} from {img.mode} to RGB for thumbnail.")
                img = img.convert('RGB')

            # Create CTkImage
            ctk_image = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))

            # Send loaded thumbnail to queue
            progress_queue.put({"type": "thumbnail_loaded", "data": {"path": file_path, "thumbnail": ctk_image}})
            loaded_count += 1
        except FileNotFoundError:
            log_message(f"File not found while loading thumbnail: {file_path}", level=logging.WARNING)
            progress_queue.put({"type": "thumbnail_error", "data": {"path": file_path, "error": "File not found"}})
            # Also update the status in the list
            progress_queue.put({"type": "update_status_list", "data": {"path": file_path, "status": STATUS_LOAD_ERROR, "details": "File not found"}})
        except Exception as e:
            log_message(f"Error creating thumbnail for {image_file.base_name}: {e}", level=logging.WARNING, exc_info=True)
            progress_queue.put({"type": "thumbnail_error", "data": {"path": file_path, "error": "Thumbnail error"}})
            # Also update the status in the list
            progress_queue.put({"type": "update_status_list", "data": {"path": file_path, "status": STATUS_THUMB_ERROR, "details": str(e)}})

    log_message(f"Thumbnail loading completed ({loaded_count}/{total_thumbs}).")
    progress_queue.put({"type": "thumbnail_load_done"})

def process_ai_thread(files_to_process: List[ImageFile], model_name: str, device: str, progress_queue: Queue, stop_event: threading.Event, stop_words: set):
    """Processes a list of ImageFile objects using AI (generates EN output)."""
    model_components = initialize_ai_model(model_name, device)
    if not model_components:
        log_message(f"AI model {model_name} initialization failed.", level=logging.CRITICAL)
        progress_queue.put({"type": "ai_done", "data": {"processed": 0, "total": len(files_to_process), "stopped": False, "error": True}})
        # Update status for all files that were supposed to be processed
        for img_file in files_to_process:
            progress_queue.put({"type": "update_status_list", "data": {"path": img_file.path, "status": STATUS_PROCESSED_ERR, "details": "AI Model Load Fail"}})
        return

    processed_count = 0
    total_files = len(files_to_process)
    log_message(f"Starting AI processing for {total_files} files with model {model_name} on {device} (EN output)")
    for i, image_file in enumerate(files_to_process):
        if stop_event.is_set():
            log_message("AI processing interrupted by user.", logging.WARNING)
            break
        try:
            file_path = image_file.path
            base_name = image_file.base_name
            # Update status to processing
            progress_queue.put({"type": "update_status_list", "data": {"path": file_path, "status": STATUS_PROCESSING_AI}})
            # Update main status bar
            progress_queue.put({"type": "status", "data": {"text": f"AI (EN): {base_name} ({i + 1}/{total_files})"}})

            # Call the generation function expecting English output
            caption, keywords = generate_caption_and_keywords(file_path, model_components, stop_words)

            if stop_event.is_set():
                break  # Check again after generation

            # Send result (even if None)
            progress_queue.put({"type": "ai_result", "data": {"path": file_path, "caption": caption, "keywords": keywords}})
            processed_count += 1
            # Update progress bar
            progress_queue.put({"type": "ai_progress", "data": {"progress": (i + 1) / total_files}})
        except Exception as e:
            log_message(f"Unexpected error in AI thread for {image_file.path}: {e}", level=logging.ERROR, exc_info=True)
            # Send error status for this specific file
            try:
                progress_queue.put({"type": "update_status_list", "data": {"path": image_file.path, "status": STATUS_PROCESSED_ERR, "details": f"Thread error: {e}"}})
            except Exception:
                pass  # Ignore if queue fails

    stopped = stop_event.is_set()
    final_msg = f"AI generation (EN) {'interrupted' if stopped else 'completed'} ({processed_count}/{total_files})."
    if not stopped and processed_count == total_files:
        final_msg += " Review/edit and save."
    log_message(final_msg, level=logging.INFO if not stopped else logging.WARNING)
    progress_queue.put({"type": "status", "data": {"text": final_msg}})
    progress_queue.put({"type": "ai_done", "data": {"processed": processed_count, "total": total_files, "stopped": stopped, "error": False}})

def save_metadata_thread(files_to_save: List[ImageFile], settings: Dict[str, Any], progress_queue: Queue, stop_event: threading.Event):
    """Saves metadata for given files in a separate thread."""
    saved_count = 0
    error_count = 0
    total_files = len(files_to_save)
    log_message(f"Starting metadata saving for {total_files} files...")
    for i, image_file in enumerate(files_to_save):
        if stop_event.is_set():
            log_message("Metadata saving interrupted by user.", logging.WARNING)
            break
        try:
            file_path = image_file.path
            base_name = image_file.base_name
            # Update status to saving
            progress_queue.put({"type": "update_status_list", "data": {"path": file_path, "status": STATUS_SAVING}})
            # Update main status bar
            progress_queue.put({"type": "status", "data": {"text": f"Saving: {base_name} ({i + 1}/{total_files})"}})

            # Get data to write (caption, keywords) from ImageFile object
            data = image_file.get_data_to_save()
            logger.debug(f"Data to save for {base_name}: Caption='{data.get('caption', '')[:30]}...', Keywords={len(data.get('keywords', []))}")

            # Call the metadata writing function
            success = write_metadata(file_path, data, settings)

            if stop_event.is_set():
                break  # Check again after writing

            # Send result back to main thread
            progress_queue.put({"type": "save_result", "data": {"path": file_path, "success": success, "error_msg": "" if success else "Metadata save error"}})
            if success:
                saved_count += 1
            else:
                error_count += 1

            # Update progress bar
            progress_queue.put({"type": "save_progress", "data": {"progress": (i + 1) / total_files}})
        except Exception as e:
            log_message(f"Unexpected error in saving thread for {image_file.path}: {e}", level=logging.ERROR, exc_info=True)
            error_count += 1
            # Send error result for this specific file
            try:
                progress_queue.put({"type": "save_result", "data": {"path": image_file.path, "success": False, "error_msg": f"Thread error: {e}"}})
            except Exception:
                pass  # Ignore if queue fails

    stopped = stop_event.is_set()
    final_msg = f"Metadata saving {'interrupted' if stopped else 'completed'}. Saved: {saved_count}/{total_files}."
    if error_count > 0:
        final_msg += f" Errors: {error_count}."
    log_message(final_msg, level=logging.INFO if error_count == 0 and not stopped else logging.WARNING)
    progress_queue.put({"type": "status", "data": {"text": final_msg}})
    progress_queue.put({"type": "save_done", "data": {"saved": saved_count, "total": total_files, "stopped": stopped}})

# --- GUI Classes ---

class ImageListItem(ctk.CTkFrame):
    """Widget representing a single image in the scrollable list."""
    def __init__(self, master, image_file: ImageFile, app: 'ImageTaggerApp', **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.image_file = image_file
        self.image_file.set_list_item_widget(self)  # Link back to widget
        self.configure(fg_color="transparent")  # Default background
        self.grid_columnconfigure(2, weight=1)  # Filename expands
        self.selected = False

        # Determine colors based on theme
        self._update_colors()

        # --- Widgets ---
        self.checkbox_var = ctk.StringVar(value="off")
        self.checkbox = ctk.CTkCheckBox(self, text="", variable=self.checkbox_var, onvalue="on", offvalue="off", width=20, command=self.app.on_list_checkbox_change)
        self.checkbox.grid(row=0, column=0, rowspan=2, padx=(5, 0), pady=2, sticky="ns")

        self.thumb_label = ctk.CTkLabel(self, text="...", width=THUMBNAIL_SIZE[0], height=THUMBNAIL_SIZE[1], fg_color="gray20", text_color="gray60")
        self.thumb_label.grid(row=0, column=1, rowspan=2, padx=5, pady=2, sticky="w")

        self.filename_label = ctk.CTkLabel(self, text=image_file.base_name, anchor="w", font=self.app.item_font)
        self.filename_label.grid(row=0, column=2, padx=5, pady=(2, 0), sticky="ew")

        self.status_label = ctk.CTkLabel(self, text=image_file.status, anchor="w", font=self.app.status_font)
        self.status_label.grid(row=1, column=2, padx=5, pady=(0, 2), sticky="ew")

        # --- Bindings ---
        # Bind click to the frame and labels (but not checkbox)
        self.bind("<Button-1>", self.on_click)
        self.thumb_label.bind("<Button-1>", self.on_click)
        self.filename_label.bind("<Button-1>", self.on_click)
        self.status_label.bind("<Button-1>", self.on_click)

        # Initial display update
        self.update_display()

    def _update_colors(self):
        """Update colors based on current theme."""
        # Get default frame color and hover color for selection highlight
        self.default_fg_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])
        # Use a slightly different color for selection, e.g., button hover color
        self.selected_fg_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["hover_color"])

    def on_click(self, event=None):
        """Handle click on the list item (selects it)."""
        # Only allow selection if no major process is running
        if not self.app.is_busy():
            self.app.select_list_item(self.image_file.path)

    def set_selected(self, selected: bool):
        """Update visual state based on selection."""
        self.selected = selected
        # Refresh colors in case theme changed
        self._update_colors()
        new_fg = self.selected_fg_color if selected else self.default_fg_color
        try:
            # Check if widget exists before configuring
            if self.winfo_exists():
                self.configure(fg_color=new_fg)
        except tk.TclError:
            # Handle cases where the widget might be destroyed during update
            pass

    def update_display(self):
        """Updates the status label text and color, and dirty indicator."""
        if not self.winfo_exists():
            return

        status_text = self.image_file.status
        status_color = STATUS_COLORS.get(status_text, "white")  # Default to white
        details = self.image_file.status_details  # TODO: Use details for tooltip?

        # Add prefix for certain statuses and dirty indicator '*'
        status_prefix_map = {
            STATUS_QUEUED_AI: "[AI] ", STATUS_PROCESSING_AI: "[AI] ",
            STATUS_PROCESSED_OK: "[AI] ", STATUS_PROCESSED_ERR: "[AI!] ",
            STATUS_SAVING: "[IO] ", STATUS_SAVED: "[OK] ",
            STATUS_SAVE_ERROR: "[ERR!] ", STATUS_LOAD_ERROR: "[ERR!] ",
            STATUS_THUMB_ERROR: "[ERR!] "
        }
        status_prefix = status_prefix_map.get(status_text, "")
        dirty_indicator = " *" if self.image_file.is_dirty else ""
        display_text = f"{status_prefix}{status_text}{dirty_indicator}"

        # Use safe_widget_update via the app instance
        self.app.safe_widget_update(self.status_label.configure, text=display_text, text_color=status_color)

    def update_thumbnail(self, ctk_image: Optional[ctk.CTkImage] = None, error: bool = False):
        """Updates the thumbnail label."""
        if not self.winfo_exists():
            return

        if error:
            self.app.safe_widget_update(self.thumb_label.configure, image=None, text="ERR", text_color="red", fg_color="gray10")
        elif ctk_image:
            self.app.safe_widget_update(self.thumb_label.configure, image=ctk_image, text="", fg_color="transparent")  # Show image, clear text/bg
        else:
            # Default placeholder state
            self.app.safe_widget_update(self.thumb_label.configure, image=None, text="...", text_color="gray60", fg_color="gray20")

class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, master, app_instance: 'ImageTaggerApp'):
        super().__init__(master)
        self.app = app_instance
        self.config = app_instance.app_settings
        self.title("Settings")
        self.geometry("550x450")
        self.transient(master)
        self.grab_set()
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)  # Allow stopwords textbox to expand

        # --- AI Settings Frame ---
        ai_frame = ctk.CTkFrame(self, fg_color="transparent")
        ai_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        ai_frame.grid_columnconfigure(1, weight=1)  # Allow combobox/radio buttons to align better

        ctk.CTkLabel(ai_frame, text="AI Model (EN output):", anchor="w").grid(row=0, column=0, padx=(0, 10), pady=5, sticky="w")
        # Ensure saved model is valid, fallback to default if not
        saved_model = self.config.get("ai_model", list(AVAILABLE_MODELS.keys())[0])
        if saved_model not in AVAILABLE_MODELS:
            saved_model = list(AVAILABLE_MODELS.keys())[0]
            self.config["ai_model"] = saved_model  # Update config if invalid model was saved
        self.model_combobox = ctk.CTkComboBox(ai_frame, values=list(AVAILABLE_MODELS.keys()), command=self.update_setting)
        self.model_combobox.set(saved_model)
        self.model_combobox.grid(row=0, column=1, columnspan=2, padx=0, pady=5, sticky="ew")

        ctk.CTkLabel(ai_frame, text="Acceleration:", anchor="w").grid(row=1, column=0, padx=(0, 10), pady=5, sticky="w")
        self.device_var = ctk.StringVar(value=self.config.get("ai_device", "auto"))
        # Disable CUDA option if not available, and reset config if CUDA was saved but isn't available now
        cuda_available = torch_available and torch.cuda.is_available()
        if self.device_var.get() == "cuda" and not cuda_available:
            self.device_var.set("auto")
            self.config["ai_device"] = "auto"  # Update config
        auto_rb = ctk.CTkRadioButton(ai_frame, text="Auto (GPU if available)", variable=self.device_var, value="auto", command=self.update_setting)
        cpu_rb = ctk.CTkRadioButton(ai_frame, text="CPU", variable=self.device_var, value="cpu", command=self.update_setting)
        cuda_rb = ctk.CTkRadioButton(ai_frame, text="GPU (CUDA)", variable=self.device_var, value="cuda", command=self.update_setting, state="normal" if cuda_available else "disabled")
        # Arrange radio buttons
        auto_rb.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        cpu_rb.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        cuda_rb.grid(row=2, column=2, padx=5, pady=5, sticky="w")  # Place CUDA next to CPU

        # --- Metadata Settings Frame ---
        meta_frame = ctk.CTkFrame(self, fg_color="transparent")
        meta_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(meta_frame, text="Write Metadata:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=3, pady=(0, 5), sticky="w")

        # Ensure saved values reflect availability, disable if not available
        iptc_on = self.config.get("write_iptc", "on") == "on" and iptc_available
        xmp_on = self.config.get("write_xmp", "on") == "on" and xmp_available
        self.write_iptc_var = ctk.StringVar(value="on" if iptc_on else "off")
        self.write_xmp_var = ctk.StringVar(value="on" if xmp_on else "off")

        self.write_iptc_check = ctk.CTkCheckBox(meta_frame, text="IPTC", variable=self.write_iptc_var, onvalue="on", offvalue="off", command=self.update_setting, state="normal" if iptc_available else "disabled")
        self.write_iptc_check.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.write_xmp_check = ctk.CTkCheckBox(meta_frame, text="XMP", variable=self.write_xmp_var, onvalue="on", offvalue="off", command=self.update_setting, state="normal" if xmp_available else "disabled")
        self.write_xmp_check.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        self.append_keywords_var = ctk.StringVar(value=self.config.get("append_keywords", "off"))
        self.append_keywords_check = ctk.CTkCheckBox(meta_frame, text="Append Keywords (don't overwrite)", variable=self.append_keywords_var, onvalue="on", offvalue="off", command=self.update_setting)
        self.append_keywords_check.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        # --- Stopwords Settings Frame ---
        stop_frame = ctk.CTkFrame(self, fg_color="transparent")
        stop_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        stop_frame.grid_rowconfigure(1, weight=1)
        stop_frame.grid_columnconfigure(0, weight=1)  # Allow textbox to expand
        ctk.CTkLabel(stop_frame, text="Custom EN Stopwords (one word per line):", anchor="w").grid(row=0, column=0, pady=(0, 5), sticky="w")
        self.custom_stopwords_textbox = ctk.CTkTextbox(stop_frame, height=80, wrap="word")
        self.custom_stopwords_textbox.grid(row=1, column=0, sticky="nsew")
        # Load saved custom stopwords
        self.custom_stopwords_textbox.insert("0.0", "\n".join(self.config.get("custom_stopwords", [])))
        # Update setting on focus out or key release? Focus out is less frequent.
        self.custom_stopwords_textbox.bind("<FocusOut>", self.update_setting)

        # --- Close Button ---
        close_button = ctk.CTkButton(self, text="Save and Close", command=self.close_dialog)
        close_button.grid(row=4, column=0, padx=20, pady=(10, 20), sticky="ew")

        # Initial update in case availability changed since last save
        self.update_setting()

    def update_setting(self, event=None):
        """Reads current widget values and updates the app's config dictionary."""
        self.config["ai_model"] = self.model_combobox.get()
        self.config["ai_device"] = self.device_var.get()
        # Only save 'on' if the library is actually available
        self.config["write_iptc"] = self.write_iptc_var.get() if iptc_available else "off"
        self.config["write_xmp"] = self.write_xmp_var.get() if xmp_available else "off"
        self.config["append_keywords"] = self.append_keywords_var.get()
        # Process custom stopwords: get text, split lines, strip, lower, unique, sort
        custom_stopwords_text = self.custom_stopwords_textbox.get("0.0", "end-1c")  # Get text excluding final newline
        self.config["custom_stopwords"] = sorted(list(set(
            line.strip().lower() for line in custom_stopwords_text.splitlines() if line.strip()
        )))
        # Trigger update in main app immediately
        self.app.update_stopwords()
        logger.debug("Settings updated in dialog.")

    def close_dialog(self):
        """Updates settings one last time, saves, and closes the dialog."""
        self.update_setting()  # Ensure latest values are captured
        self.app.save_settings()  # Trigger save in main app
        logger.info("Settings dialog closed and saved.")
        self.grab_release()
        self.destroy()

class AboutDialog(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("About")
        self.geometry("400x250")
        self.transient(master)
        self.grab_set()
        self.grid_columnconfigure(0, weight=1)

        title_label = ctk.CTkLabel(self, text=f"AI Image Tagger v{APP_VERSION}", font=ctk.CTkFont(size=16, weight="bold"))
        title_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        info_text = ("A simple desktop application for generating\n"
                     "English captions and keywords for images\n"
                     "using AI (Hugging Face Transformers) and writing\n"
                     "them to IPTC/XMP metadata.\n\n"
                     f"Log file: {LOG_FILENAME}\n"
                     f"Config file: {CONFIG_FILE}")
        info_label = ctk.CTkLabel(self, text=info_text, justify=tk.LEFT)
        info_label.grid(row=1, column=0, padx=20, pady=5)

        close_button = ctk.CTkButton(self, text="Close", command=self.destroy)
        close_button.grid(row=2, column=0, padx=20, pady=(10, 20))

class ListPanel(ctk.CTkFrame):
    """Panel containing the image list and related controls."""
    def __init__(self, master, app_instance: 'ImageTaggerApp', **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.app = app_instance
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        # Store references to list item widgets, keyed by path
        self.list_item_widgets: Dict[str, ImageListItem] = {}

        # --- Filter Frame ---
        filter_frame = ctk.CTkFrame(self, fg_color="transparent")
        filter_frame.grid(row=0, column=0, padx=0, pady=(0, 5), sticky="ew")
        self.search_entry = ctk.CTkEntry(filter_frame, placeholder_text="Filter by name...")
        self.search_entry.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        self.search_entry.bind("<KeyRelease>", lambda e: self.app.filter_image_list())
        self.clear_search_button = ctk.CTkButton(filter_frame, text="X", width=25, command=self.app.clear_filter)
        self.clear_search_button.pack(side=tk.LEFT)

        # --- Select Buttons Frame ---
        select_buttons_frame = ctk.CTkFrame(self, fg_color="transparent")
        select_buttons_frame.grid(row=1, column=0, padx=0, pady=(0, 5), sticky="ew")
        self.select_all_button = ctk.CTkButton(select_buttons_frame, text="Select All", command=lambda: self.app.toggle_select_all(True), state="disabled", width=100)
        self.select_all_button.pack(side=tk.LEFT, padx=5)
        self.select_none_button = ctk.CTkButton(select_buttons_frame, text="Deselect All", command=lambda: self.app.toggle_select_all(False), state="disabled", width=100)
        self.select_none_button.pack(side=tk.LEFT, padx=5)

        # --- Scrollable List Frame ---
        self.scrollable_frame = ctk.CTkScrollableFrame(self, label_text="Found Images (0)")
        self.scrollable_frame.grid(row=2, column=0, padx=0, pady=0, sticky="nsew")
        self.scrollable_frame.grid_columnconfigure(0, weight=1)  # Ensure items inside expand

        # --- Log Textbox Frame ---
        self.log_textbox = ctk.CTkTextbox(self, height=100, state="disabled", wrap="word", font=ctk.CTkFont(size=10))
        self.log_textbox.grid(row=3, column=0, padx=0, pady=(5, 0), sticky="ew")

    def add_item(self, image_file: ImageFile):
        """Creates and adds an ImageListItem widget to the scrollable frame."""
        if image_file.path not in self.list_item_widgets:
            list_item = ImageListItem(self.scrollable_frame, image_file, self.app)
            list_item.grid(sticky="ew", padx=5, pady=2)
            self.list_item_widgets[image_file.path] = list_item
            self.update_label_count()  # Update count when adding
        else:
            logger.warning(f"Attempted to add duplicate list item for {image_file.path}")

    def clear_list(self):
        """Removes all ImageListItem widgets and clears the reference dictionary."""
        widgets_to_destroy = list(self.list_item_widgets.values())
        for widget in widgets_to_destroy:
            if widget.winfo_exists():
                widget.destroy()
        self.list_item_widgets.clear()
        self.update_label_count()  # Reset count

    def update_label_count(self):
        """Updates the scrollable frame's label with the current item count."""
        total_count = len(self.app.image_files)  # Get total from app
        # Count visible items (those not removed by filter)
        visible_count = len([w for w in self.list_item_widgets.values() if w.winfo_ismapped()])
        filter_active = bool(self.search_entry.get())

        if filter_active:
            label = f"Found Images ({visible_count}/{total_count} - Filter Active)"
        else:
            label = f"Found Images ({total_count})"
        self.app.safe_widget_update(self.scrollable_frame.configure, label_text=label)

    def filter_items(self, search_term: str):
        """Shows or hides items based on the search term."""
        search_lower = search_term.lower()
        visible_count = 0
        for path, widget in self.list_item_widgets.items():
            if widget.winfo_exists():
                base_name_lower = widget.image_file.base_name.lower()
                is_visible = search_lower in base_name_lower
                if is_visible:
                    if not widget.winfo_ismapped():
                        widget.grid()  # Show if hidden
                    visible_count += 1
                else:
                    if widget.winfo_ismapped():
                        widget.grid_remove()  # Hide if shown

        # Update label after filtering
        self.update_label_count()

class DetailPanel(ctk.CTkFrame):
    """Panel containing the large preview and metadata details."""
    def __init__(self, master, app_instance: 'ImageTaggerApp', **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.app = app_instance
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # Detail frame expands

        # --- Preview Label ---
        self.preview_label = ctk.CTkLabel(self, text="", width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1], fg_color="gray15")
        self.preview_label.grid(row=0, column=0, padx=10, pady=(0, 10), sticky="n")

        # --- Detail Frame (holds metadata fields) ---
        detail_frame = ctk.CTkFrame(self)
        detail_frame.grid(row=1, column=0, padx=10, pady=0, sticky="nsew")
        detail_frame.grid_columnconfigure(0, weight=1)  # Allow textboxes to expand horizontally
        # Make the last row (keywords) expand vertically
        detail_frame.grid_rowconfigure(6, weight=1)

        # --- Filename Title ---
        self.filename_label = ctk.CTkLabel(detail_frame, text="Select an image", font=self.app.detail_title_font, anchor="w")
        self.filename_label.grid(row=0, column=0, padx=10, pady=(5, 10), sticky="ew")

        # --- Existing Metadata Section ---
        ctk.CTkLabel(detail_frame, text="EXISTING METADATA", font=self.app.detail_label_font, anchor="w").grid(row=1, column=0, padx=10, pady=(5, 2), sticky="w")
        self.existing_caption = ctk.CTkTextbox(detail_frame, height=60, font=self.app.detail_font, state="disabled", wrap="word", fg_color="gray20")
        self.existing_caption.grid(row=2, column=0, padx=10, pady=(0, 5), sticky="ew")
        self.existing_keywords = ctk.CTkTextbox(detail_frame, height=70, font=self.app.detail_font, state="disabled", wrap="word", fg_color="gray20")
        self.existing_keywords.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")

        # --- Generated/Editable Metadata Section ---
        ctk.CTkLabel(detail_frame, text="GENERATED/EDITABLE METADATA (AI generates in ENGLISH)", font=self.app.detail_label_font, anchor="w").grid(row=4, column=0, padx=10, pady=(10, 2), sticky="w")
        self.generated_caption = ctk.CTkTextbox(detail_frame, height=80, font=self.app.detail_font, state="disabled", wrap="word")
        self.generated_caption.grid(row=5, column=0, padx=10, pady=(0, 5), sticky="ew")
        self.generated_keywords = ctk.CTkTextbox(detail_frame, height=100, font=self.app.detail_font, state="disabled", wrap="word")  # Will expand vertically due to rowconfigure
        self.generated_keywords.grid(row=6, column=0, padx=10, pady=(0, 10), sticky="nsew")  # Use nsew to expand

        # --- Bind edits ---
        # Use lambda to call app's methods
        self.generated_caption.bind("<FocusOut>", lambda e: self.app.on_detail_edit())
        self.generated_caption.bind("<KeyRelease>", lambda e: self.app.on_detail_edit_key())  # Update on key release
        self.generated_keywords.bind("<FocusOut>", lambda e: self.app.on_detail_edit())
        self.generated_keywords.bind("<KeyRelease>", lambda e: self.app.on_detail_edit_key())  # Update on key release

        # --- Save Current Button ---
        self.save_current_button = ctk.CTkButton(detail_frame, text="Save Current Image", command=self.app.save_current_image, state="disabled")
        self.save_current_button.grid(row=7, column=0, padx=10, pady=(5, 10), sticky="ew")

    def clear_details(self):
        """Clears all widgets within the detail panel."""
        logger.debug("Clearing detail panel widgets.")
        self.app.safe_widget_update(self.preview_label.configure, image=None, text="")
        # CTkImage handles its own references, no need for self.preview_label.image = None
        self.app.safe_widget_update(self.filename_label.configure, text="Select an image from the list")

        widgets_to_clear = [
            self.existing_caption, self.existing_keywords,
            self.generated_caption, self.generated_keywords
        ]
        for widget in widgets_to_clear:
            # Use safe_widget_update from the app instance
            self.app.safe_widget_update(widget.configure, state="normal")
            # CTkTextbox uses "0.0" for start index
            self.app.safe_widget_update(widget.delete, "0.0", "end")
            # Set back to disabled (except for generated fields which might be enabled later)
            if widget in [self.existing_caption, self.existing_keywords]:
                self.app.safe_widget_update(widget.configure, state="disabled")
            else:
                # Keep generated fields disabled initially after clearing
                self.app.safe_widget_update(widget.configure, state="disabled")

        # Also reset the save button state within the detail panel
        self.app.safe_widget_update(self.save_current_button.configure, state="disabled")

    # --- Main Application Class ---
class ImageTaggerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        log_message(f"Starting AI Image Tagger v{APP_VERSION}", level=logging.INFO)
        self.title(f"AI Image Tagger & Describer v{APP_VERSION}")
        self.app_settings = self.load_settings()
        self.geometry(self.app_settings.get("window_geometry", "1350x850"))
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # --- Data & State ---
        self.selected_folder: Optional[str] = self.app_settings.get("last_folder")
        self.image_files: Dict[str, ImageFile] = {}
        self.currently_selected_file: Optional[str] = None
        self.combined_stopwords = set()
        self._detail_edit_pending = False  # Flag for debouncing key release edits

        # --- Threading & Queue ---
        self.progress_queue = Queue()
        self.stop_event = threading.Event()
        self.file_load_thread: Optional[threading.Thread] = None
        self.thumb_load_thread: Optional[threading.Thread] = None
        self.ai_process_thread: Optional[threading.Thread] = None
        self.save_thread: Optional[threading.Thread] = None

        # --- Font Definitions ---
        self._define_fonts()
        self.update_stopwords()  # Initial calculation

        # --- Build UI ---
        self._setup_ui()  # This creates self.list_panel and self.detail_panel

        # --- Final Setup ---
        self.update_folder_display()
        if self.selected_folder and os.path.isdir(self.selected_folder):
            log_message(f"Automatically loading files from: {self.selected_folder}", logging.INFO)
            self.start_load_files()
        else:
            self.update_action_buttons_state()  # Set initial button states

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.after(100, self.check_queue)  # Start queue checker

    def _define_fonts(self):
        size = 11 if platform.system() == "Windows" else 12
        self.item_font = ctk.CTkFont(family="Segoe UI", size=size)
        self.status_font = ctk.CTkFont(family="Segoe UI", size=size - 1)
        self.detail_font = ctk.CTkFont(family="Segoe UI", size=size + 1)
        self.detail_label_font = ctk.CTkFont(family="Segoe UI", size=size, weight="bold")
        self.detail_title_font = ctk.CTkFont(family="Segoe UI", size=size + 2, weight="bold")

    def _setup_ui(self):
        """Creates main UI layout, panels, menus, and status bar."""
        self.grid_columnconfigure(0, weight=5)
        self.grid_columnconfigure(1, weight=5, minsize=450)
        self.grid_rowconfigure(1, weight=1)  # Main content row expands

        self._create_menus()

        # --- Top Bar ---
        self.top_bar = ctk.CTkFrame(self, fg_color="transparent")
        self.top_bar.grid(row=0, column=0, columnspan=2, padx=10, pady=(5, 5), sticky="ew")  # Reduced top padding
        self.select_folder_button = ctk.CTkButton(self.top_bar, text="Select Folder", width=120, command=self.select_folder)
        self.select_folder_button.pack(side=tk.LEFT, padx=(0, 5), pady=0)  # Reduced pady
        self.folder_path_label = ctk.CTkLabel(self.top_bar, text="...", anchor="w", wraplength=500)
        self.folder_path_label.pack(side=tk.LEFT, padx=5, pady=0, fill=tk.X, expand=True)
        self.load_files_button = ctk.CTkButton(self.top_bar, text="Reload", width=120, command=self.start_load_files, state="disabled")
        self.load_files_button.pack(side=tk.LEFT, padx=5, pady=0)
        # Settings button moved to menu

        # --- Main Panels ---
        # Create instances of the panel classes
        self.list_panel = ListPanel(self, self)
        self.list_panel.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        self.detail_panel = DetailPanel(self, self)
        self.detail_panel.grid(row=1, column=1, padx=(0, 10), pady=5, sticky="nsew")

        # --- Status Bar ---
        self._create_status_bar()

    def _create_menus(self):
        """Creates the application menu bar."""
        self.menu_bar = Menu(self)
        self.config(menu=self.menu_bar)  # Attach menu to main window

        # File Menu
        file_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Select Folder...", command=self.select_folder)
        # Add state management for reload based on folder selection?
        self.reload_menu_item_index = file_menu.add_command(label="Reload Folder", command=self.start_load_files, state="disabled")
        file_menu.add_separator()
        file_menu.add_command(label="Settings...", command=self.open_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

        # Help Menu
        help_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about_dialog)

    def _create_status_bar(self):
        """Creates the bottom status bar area."""
        self.status_bar_frame = ctk.CTkFrame(self, height=60)  # Slightly taller for buttons+progress+text
        self.status_bar_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")
        self.status_bar_frame.grid_columnconfigure((0, 1, 2), weight=1)  # Buttons expand

        # Action Buttons (within status bar frame)
        self.process_button = ctk.CTkButton(self.status_bar_frame, text="Generate Selected (EN)", command=self.start_ai_processing, state="disabled")
        self.process_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.stop_button = ctk.CTkButton(self.status_bar_frame, text="Stop Process", command=self.stop_current_process, state="disabled", fg_color="#D32F2F", hover_color="#B71C1C")
        self.stop_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.save_selected_button = ctk.CTkButton(self.status_bar_frame, text="Save Selected", command=self.start_save_selected, state="disabled")
        self.save_selected_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        # Progress Bar
        self.progress_bar = ctk.CTkProgressBar(self.status_bar_frame, orientation="horizontal", mode='determinate')
        self.progress_bar.grid(row=1, column=0, columnspan=3, padx=5, pady=(5, 2), sticky="ew")
        self.progress_bar.set(0)
        # Status Label
        self.status_label = ctk.CTkLabel(self.status_bar_frame, text="Ready.", anchor="w", wraplength=1000)  # Adjust wraplength as needed
        self.status_label.grid(row=2, column=0, columnspan=3, padx=5, pady=(0, 5), sticky="ew")

    # --- Configuration & Settings ---
    def load_settings(self) -> Dict[str, Any]:
        defaults = {
            "last_folder": "",
            "ai_model": list(AVAILABLE_MODELS.keys())[0],
            "ai_device": "auto",
            "write_iptc": "on" if iptc_available else "off",
            "write_xmp": "on" if xmp_available else "off",
            "append_keywords": "off",
            "custom_stopwords": [],
            "window_geometry": "1350x850"
        }
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    # Ensure loaded settings don't override availability checks
                    if not iptc_available:
                        loaded_settings["write_iptc"] = "off"
                    if not xmp_available:
                        loaded_settings["write_xmp"] = "off"
                    defaults.update(loaded_settings)
                logger.info(f"Configuration file loaded: {CONFIG_FILE}")
            else:
                logger.info(f"Configuration file not found, using default settings: {CONFIG_FILE}")
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error in configuration file: {e}. Using default settings.")
        except Exception as e:
            logger.warning(f"Error loading configuration file: {e}", exc_info=True)

        # Validate loaded/default settings
        if defaults["ai_model"] not in AVAILABLE_MODELS:
            logger.warning(f"Invalid AI model '{defaults['ai_model']}' in configuration, falling back to default.")
            defaults["ai_model"] = list(AVAILABLE_MODELS.keys())[0]
        if defaults["ai_device"] not in ["auto", "cpu", "cuda"]:
            logger.warning(f"Invalid AI device '{defaults['ai_device']}' in configuration, falling back to 'auto'.")
            defaults["ai_device"] = "auto"
        # Ensure boolean-like settings are 'on' or 'off'
        for key in ["write_iptc", "write_xmp", "append_keywords"]:
            if defaults[key] not in ["on", "off"]:
                defaults[key] = "off"

        return defaults

    def save_settings(self):
        self.app_settings["last_folder"] = self.selected_folder
        try:
            # Only save geometry if the window is in a normal state (not minimized/maximized)
            if self.state() == 'normal':
                self.app_settings["window_geometry"] = self.geometry()
        except tk.TclError:
            logger.debug("Failed to get window geometry for saving (window probably closed).")
            pass  # Ignore if window doesn't exist
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.app_settings, f, indent=4, ensure_ascii=False)
            logger.info(f"Settings saved to: {CONFIG_FILE}")
        except Exception as e:
            log_message(f"Error saving settings: {e}", logging.ERROR, exc_info=True)

    def update_stopwords(self):
        custom = set(str(sw).strip().lower() for sw in self.app_settings.get("custom_stopwords", []) if str(sw).strip())
        self.combined_stopwords = DEFAULT_STOP_WORDS_EN.union(custom)
        logger.debug(f"EN Stopwords updated: {len(self.combined_stopwords)} total (including {len(custom)} custom)")

    def open_settings(self):
        if self.is_busy():
            log_message("Cannot open settings during an active process.", logging.WARNING)
            messagebox.showwarning("Process Active", "Cannot open settings while a process is running.", parent=self)
            return
        # Check if dialog already exists and bring it to front
        if hasattr(self, 'settings_dialog') and self.settings_dialog and self.settings_dialog.winfo_exists():
            self.settings_dialog.lift()
            self.settings_dialog.focus()
            return
        self.settings_dialog = SettingsDialog(self, self)

    def show_about_dialog(self):
        # Check if dialog already exists and bring it to front
        if hasattr(self, 'about_dialog') and self.about_dialog and self.about_dialog.winfo_exists():
            self.about_dialog.lift()
            self.about_dialog.focus()
            return
        self.about_dialog = AboutDialog(self)

    # --- File Loading & List Management ---
    def select_folder(self):
        if self.is_busy():
            log_message("Cannot change folder during an active process.", logging.WARNING)
            messagebox.showwarning("Process Active", "Cannot change folder while a process is running.", parent=self)
            return
        idir = self.selected_folder if self.selected_folder and os.path.isdir(self.selected_folder) else "/"
        folder = fd.askdirectory(initialdir=idir, parent=self, title="Select Folder with Images")
        if folder:
            self.selected_folder = folder
            self.app_settings["last_folder"] = folder  # Save selection immediately
            self.update_folder_display()
            self.clear_file_list()  # Clear previous content
            self.clear_detail_panel()  # Clear detail panel
            log_message(f"Folder selected: {folder}", logging.INFO)
            self.start_load_files()  # Start loading automatically
        else:
            log_message("Folder selection canceled.", logging.INFO)

    def update_folder_display(self):
        path = self.selected_folder or "No folder selected"
        self.safe_widget_update(self.folder_path_label.configure, text=path)
        can_load = bool(self.selected_folder and os.path.isdir(self.selected_folder))
        btn_text = "Reload" if can_load else "Load Files"  # Button text might not be needed if menu is primary
        self.safe_widget_update(self.load_files_button.configure, text=btn_text, state="normal" if can_load else "disabled")
        # Update menu item state
        menu_state = "normal" if can_load else "disabled"
        if hasattr(self, 'menu_bar') and hasattr(self, 'reload_menu_item_index'):
            try:
                self.menu_bar.entryconfigure(self.reload_menu_item_index, state=menu_state)
            except tk.TclError:
                pass  # Ignore if menu doesn't exist yet

    def start_load_files(self):
        if not self.selected_folder or not os.path.isdir(self.selected_folder):
            log_message("Invalid or no folder selected.", logging.WARNING)
            messagebox.showwarning("Invalid Folder", "Please select a valid folder containing images.", parent=self)
            return
        if self.is_busy():
            log_message("Cannot load files, a process is already running.", logging.WARNING)
            messagebox.showwarning("Process Active", "Cannot load files while a process is running.", parent=self)
            return

        # Clear previous state
        self.clear_file_list()
        self.clear_detail_panel()

        # Start loading process
        self.update_status("Scanning folder and loading metadata...")
        self.progress_bar.set(0)  # Reset progress bar
        self.stop_event.clear()  # Clear stop flag for the new process
        self.update_action_buttons_state()  # Disable buttons during load

        # Start the thread
        self.file_load_thread = threading.Thread(
            target=load_files_and_metadata_thread,
            args=(self.selected_folder, self.progress_queue, self.stop_event),
            name="FileMetaLoader",
            daemon=True  # Daemon threads exit automatically if main app exits
        )
        self.file_load_thread.start()

    def start_load_thumbnails(self, files_to_load: List[ImageFile]):
        if self.is_busy("ThumbnailLoader"):
            log_message("Thumbnail loading is already in progress.", logging.WARNING)
            return
        if not files_to_load:
            logger.debug("No files to load thumbnails for.")
            # If metadata loading finished but no files, update status
            if not self.is_busy("FileMetaLoader"):
                self.update_status("No supported images found.", "info")
                self.update_action_buttons_state()
            return

        self.update_status(f"Loading {len(files_to_load)} thumbnails...")
        # Don't reset progress bar here, let metadata load finish first if running
        self.update_action_buttons_state()  # Update buttons (Stop might become active)

        # Start the thread
        self.thumb_load_thread = threading.Thread(
            target=load_thumbnails_thread,
            args=(files_to_load, self.progress_queue),
            name="ThumbnailLoader",
            daemon=True
        )
        self.thumb_load_thread.start()

    def clear_file_list(self):
        """Clears the internal data store and the list panel UI."""
        self.currently_selected_file = None
        self.image_files.clear()  # Clear the main data dictionary
        # Delegate clearing the UI to ListPanel
        if hasattr(self, 'list_panel') and self.list_panel:
            self.list_panel.clear_list()  # Call ListPanel's method
        # Update buttons after clearing
        self.update_action_buttons_state()
        log_message("File list cleared.", logging.DEBUG)

    # add_list_item is handled by the queue processing calling list_panel.add_item
    # update_list_label_count is handled by list_panel internally

    def filter_image_list(self):
        """Filters the items displayed in the list panel."""
        if hasattr(self, 'list_panel') and self.list_panel:
            search_term = self.list_panel.search_entry.get()
            self.list_panel.filter_items(search_term)  # Delegate to ListPanel

    def clear_filter(self):
        """Clears the search filter in the list panel."""
        if hasattr(self, 'list_panel') and self.list_panel:
            self.list_panel.search_entry.delete(0, tk.END)
            self.filter_image_list()  # Re-apply empty filter to show all

    def select_list_item(self, file_path: str):
        """Handles selection of an item in the list."""
        if self.is_busy():
            log_message("Item selection not allowed during an active process.", logging.DEBUG)
            return
        if file_path not in self.image_files:
            log_message(f"Attempted to select unknown file: {file_path}", logging.WARNING)
            return
        if file_path == self.currently_selected_file:
            log_message(f"File {os.path.basename(file_path)} is already selected.", logging.DEBUG)
            return  # Already selected

        # --- Deselect previous item ---
        if self.currently_selected_file:
            # Update data from detail panel before switching
            self.update_image_data_from_detail(self.currently_selected_file, force_update=False)  # Update internal data if edited
            # Get the widget for the previously selected item
            prev_list_widget = self.list_panel.list_item_widgets.get(self.currently_selected_file)
            if prev_list_widget:
                prev_list_widget.set_selected(False)  # Update UI

        # --- Select new item ---
        self.currently_selected_file = file_path
        image_file = self.image_files[file_path]
        # Get the widget for the newly selected item
        new_list_widget = self.list_panel.list_item_widgets.get(file_path)
        if new_list_widget:
            new_list_widget.set_selected(True)  # Update UI

        # --- Update Detail Panel ---
        self.show_large_preview(file_path)
        self.display_details(image_file)

        # Update button states based on new selection
        self.update_action_buttons_state()
        log_message(f"File selected: {image_file.base_name}", logging.DEBUG)

    def toggle_select_all(self, select_state: bool):
        """Selects or deselects all currently visible items in the list."""
        if self.is_busy():
            return
        target = "on" if select_state else "off"
        count = 0
        # Iterate through the widgets stored in the ListPanel's dictionary
        for widget in self.list_panel.list_item_widgets.values():
            # Only toggle checkboxes for items that are currently visible (not filtered out)
            if widget.winfo_exists() and widget.winfo_ismapped():
                widget.checkbox_var.set(target)
                count += 1
        log_message(f"{'Selected' if select_state else 'Deselected'} {count} visible items.")
        # Update button states after changing selection
        self.on_list_checkbox_change()

    def get_selected_files(self) -> List[ImageFile]:
        """Returns a list of ImageFile objects for currently checked items."""
        selected = []
        # Iterate through the widgets stored in the ListPanel's dictionary
        for widget in self.list_panel.list_item_widgets.values():
            # Check if the widget exists and its checkbox is 'on'
            if widget.winfo_exists() and widget.checkbox_var.get() == "on":
                # Ensure the corresponding ImageFile object exists in the main dictionary
                if widget.image_file.path in self.image_files:
                    selected.append(widget.image_file)
        return selected

    def on_list_checkbox_change(self):
        """Called when any checkbox in the list changes state."""
        # Simply update the main action buttons state
        self.update_action_buttons_state()

    # --- Detail Panel Management ---
    def clear_detail_panel(self):
        """Clears the content of the detail panel."""
        self.currently_selected_file = None  # Also clear app's selection tracker
        if hasattr(self, 'detail_panel') and self.detail_panel:
            self.detail_panel.clear_details()  # Call the method within DetailPanel
        # Update buttons as selection is now gone
        self.update_action_buttons_state()

    def show_large_preview(self, file_path: str):
        """Loads and displays the large preview image in the detail panel."""
        if not hasattr(self, 'detail_panel') or not self.detail_panel:
            return
        preview_widget = self.detail_panel.preview_label
        if not preview_widget.winfo_exists():
            return

        try:
            img = Image.open(file_path)
            orientation = 1
            try:
                exif_data = img.getexif()
                orientation = exif_data.get(0x0112, 1) if exif_data else 1
            except Exception as exif_err:
                logger.debug(f"EXIF read error for preview {os.path.basename(file_path)}: {exif_err}")

            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)

            img.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')

            ctk_image = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
            self.safe_widget_update(preview_widget.configure, image=ctk_image, text="")
        except FileNotFoundError:
            log_message(f"File not found for large preview: {file_path}", logging.WARNING)
            self.safe_widget_update(preview_widget.configure, image=None, text="File not found", text_color="red")
        except Exception as e:
            log_message(f"Error loading large preview for {os.path.basename(file_path)}: {e}", logging.WARNING, exc_info=True)
            self.safe_widget_update(preview_widget.configure, image=None, text="Preview error", text_color="red")

    def display_details(self, image_file: ImageFile):
        """Populates the detail panel with data from the ImageFile object."""
        if not hasattr(self, 'detail_panel') or not self.detail_panel:
            return
        dp = self.detail_panel  # Shorthand

        # Update filename label
        self.safe_widget_update(dp.filename_label.configure, text=image_file.base_name)

        # --- Display Existing Metadata ---
        ex_caption = image_file.existing_meta.get("caption", "") or ""
        ex_keywords = image_file.existing_meta.get("keywords", []) or []
        ex_source = image_file.existing_meta.get("source", "N/A")
        # Update existing caption textbox
        self.safe_widget_update(dp.existing_caption.configure, state="normal")
        self.safe_widget_update(dp.existing_caption.delete, "0.0", "end")
        self.safe_widget_update(dp.existing_caption.insert, "0.0", ex_caption)
        self.safe_widget_update(dp.existing_caption.configure, state="disabled")
        # Update existing keywords textbox
        self.safe_widget_update(dp.existing_keywords.configure, state="normal")
        self.safe_widget_update(dp.existing_keywords.delete, "0.0", "end")
        self.safe_widget_update(dp.existing_keywords.insert, "0.0", "\n".join(ex_keywords))
        self.safe_widget_update(dp.existing_keywords.configure, state="disabled")
        # TODO: Display ex_source somewhere? Maybe tooltip or small label?

        # --- Display Generated/Editable Metadata ---
        # Get caption/keywords to display (edited takes precedence over generated)
        display_caption = image_file.get_display_caption()
        display_keywords = image_file.get_display_keywords()
        # Determine if fields should be editable
        can_edit = image_file.status in [STATUS_PROCESSED_OK, STATUS_SAVED, STATUS_SAVE_ERROR, STATUS_READY, STATUS_PROCESSED_ERR] and not self.is_busy("AIProcessor")
        # Update generated/edited caption textbox
        self.safe_widget_update(dp.generated_caption.configure, state="normal")
        self.safe_widget_update(dp.generated_caption.delete, "0.0", "end")
        self.safe_widget_update(dp.generated_caption.insert, "0.0", display_caption)
        self.safe_widget_update(dp.generated_caption.configure, state="normal" if can_edit else "disabled")
        # Update generated/edited keywords textbox
        self.safe_widget_update(dp.generated_keywords.configure, state="normal")
        self.safe_widget_update(dp.generated_keywords.delete, "0.0", "end")
        self.safe_widget_update(dp.generated_keywords.insert, "0.0", "\n".join(display_keywords))
        self.safe_widget_update(dp.generated_keywords.configure, state="normal" if can_edit else "disabled")

        # Update dirty indicator (e.g., change border color or add label)
        self.update_detail_panel_dirty_indicator()
        # Update button states (especially the "Save Current" button)
        self.update_action_buttons_state()

    def update_detail_panel_dirty_indicator(self):
        """Updates a visual cue in the detail panel if the current image has unsaved changes."""
        if not hasattr(self, 'detail_panel') or not self.detail_panel:
            return
        if not self.currently_selected_file or self.currently_selected_file not in self.image_files:
            return

        image_file = self.image_files[self.currently_selected_file]
        is_dirty = image_file.is_dirty
        # Example: Change border color of the editable fields
        border_color = "yellow" if is_dirty else ctk.ThemeManager.theme["CTkTextbox"]["border_color"]
        self.safe_widget_update(self.detail_panel.generated_caption.configure, border_color=border_color)
        self.safe_widget_update(self.detail_panel.generated_keywords.configure, border_color=border_color)
        # Or update the filename label
        # filename_text = f"{image_file.base_name}{' *' if is_dirty else ''}"
        # self.safe_widget_update(self.detail_panel.filename_label.configure, text=filename_text)

    def on_detail_edit_key(self):
        """Called on KeyRelease in editable fields. Debounces the update."""
        if not self._detail_edit_pending:
            self._detail_edit_pending = True
            # Schedule the actual update after a short delay (e.g., 300ms)
            self.after(300, self.perform_detail_update)

    def on_detail_edit(self):
        """Called on FocusOut from editable fields. Updates immediately."""
        # Cancel any pending key release update
        if self._detail_edit_pending:
            # How to cancel an 'after' job? Need to store the ID.
            # For simplicity, let perform_detail_update handle the flag reset.
            pass
        self.perform_detail_update()

    def perform_detail_update(self):
        """Reads data from detail panel and updates the ImageFile object."""
        self._detail_edit_pending = False  # Reset flag
        if self.currently_selected_file:
            self.update_image_data_from_detail(self.currently_selected_file, force_update=True)

    def update_image_data_from_detail(self, file_path: str, force_update: bool = True):
        """Reads data from detail panel widgets and updates the corresponding ImageFile object."""
        if not file_path or file_path not in self.image_files:
            return
        if not hasattr(self, 'detail_panel') or not self.detail_panel:
            return

        image_file = self.image_files[file_path]
        dp = self.detail_panel

        # Only read if the widgets are enabled (editable) or if forced
        can_read = dp.generated_caption.cget("state") == "normal" or force_update
        if can_read:
            try:
                # Check if widgets exist before getting text
                if dp.generated_caption.winfo_exists() and dp.generated_keywords.winfo_exists():
                    caption_text = dp.generated_caption.get("0.0", "end-1c")
                    keywords_text = dp.generated_keywords.get("0.0", "end-1c")
                    keywords_list = [line.strip() for line in keywords_text.splitlines() if line.strip()]

                    # Call the ImageFile method to update its edited state and dirty flag
                    image_file.update_edited(caption_text, keywords_list)

                    # Update UI elements if the file is still selected
                    if self.currently_selected_file == file_path:
                        self.update_detail_panel_dirty_indicator()
                        self.update_action_buttons_state()  # Update save buttons

                    logger.debug(f"Data for {image_file.base_name} updated from detail.")
                else:
                    logger.debug(f"Detail widgets for {image_file.base_name} do not exist, skipping read.")

            except tk.TclError as e:
                # Handle cases where widget might be destroyed during access
                logger.warning(f"Error reading detail widget for {image_file.base_name} (widget gone?): {e}")
            except Exception as e:
                log_message(f"Unexpected error reading detail widget: {e}", level=logging.ERROR, exc_info=True)

    # --- Processing & Saving ---
    def start_ai_processing(self):
        """Starts the AI processing thread for selected files."""
        if self.is_busy():
            log_message("Cannot start AI processing, a process is already running.", logging.WARNING)
            messagebox.showwarning("Process Active", "Cannot start AI processing while a process is running.", parent=self)
            return

        files_to_process = self.get_selected_files()
        if not files_to_process:
            log_message("No files selected for AI processing.", logging.WARNING)
            messagebox.showinfo("No Selection", "Please select files to generate metadata.", parent=self)
            return

        # Save any pending edits in the detail panel before starting
        if self.currently_selected_file:
            self.update_image_data_from_detail(self.currently_selected_file, force_update=True)

        # Get AI settings
        model_name = self.app_settings.get("ai_model")
        device = get_device(self.app_settings.get("ai_device", "auto"))

        # Update UI and start thread
        self.update_status(f"Starting AI (EN) for {len(files_to_process)} files ({model_name} on {device})...")
        self.progress_bar.set(0)
        self.stop_event.clear()
        for f in files_to_process:
            f.update_status(STATUS_QUEUED_AI)  # Update status in ImageFile object (triggers list item update)
        self.update_action_buttons_state()  # Disable buttons, enable Stop
        self.ai_process_thread = threading.Thread(
            target=process_ai_thread,
            args=(files_to_process, model_name, device, self.progress_queue, self.stop_event, self.combined_stopwords),
            name="AIProcessor",
            daemon=True
        )
        self.ai_process_thread.start()

    def start_save_selected(self):
        """Starts the metadata saving thread for selected files with changes."""
        if self.is_busy():
            log_message("Cannot save metadata, a process is already running.", logging.WARNING)
            messagebox.showwarning("Process Active", "Cannot save metadata while a process is running.", parent=self)
            return

        # Save any pending edits in the detail panel first
        if self.currently_selected_file:
            self.update_image_data_from_detail(self.currently_selected_file, force_update=True)

        # Get selected files that are actually dirty (have changes)
        files_to_save = [f for f in self.get_selected_files() if f.is_dirty]
        if not files_to_save:
            log_message("No selected files with unsaved changes.", logging.INFO)
            messagebox.showinfo("No Changes", "No selected files have unsaved changes.", parent=self)
            return

        # Get current save settings
        settings = self._get_current_save_settings()
        if not settings["write_iptc"] and not settings["write_xmp"]:
            log_message("No metadata formats selected for saving in settings.", logging.WARNING)
            messagebox.showwarning("No Save Format", "Please select at least one metadata format (IPTC or XMP) in settings.", parent=self)
            return

        # Initiate the save thread
        self._initiate_save_thread(files_to_save, settings)

    def save_current_image(self):
        """Saves metadata only for the image currently shown in the detail panel, if it has changes."""
        if self.is_busy():
            log_message("Cannot save image, a process is already running.", logging.WARNING)
            messagebox.showwarning("Process Active", "Cannot save image while a process is running.", parent=self)
            return
        if not self.currently_selected_file:
            log_message("No image selected in detail panel.", logging.WARNING)
            return  # Should not happen if button is enabled correctly

        img_file = self.image_files.get(self.currently_selected_file)
        if not img_file:
            log_message("Data for currently selected image not found.", logging.ERROR)
            return

        # Ensure latest edits from textboxes are in the ImageFile object
        self.update_image_data_from_detail(self.currently_selected_file, force_update=True)

        if not img_file.is_dirty:
            log_message(f"{img_file.base_name} has no unsaved changes.", logging.INFO)
            # Optionally show info message to user
            # messagebox.showinfo("No Changes", f"{img_file.base_name} has no unsaved changes.", parent=self)
            return

        # Get current save settings
        settings = self._get_current_save_settings()
        if not settings["write_iptc"] and not settings["write_xmp"]:
            log_message("No metadata formats selected for saving in settings.", logging.WARNING)
            messagebox.showwarning("No Save Format", "Please select at least one metadata format (IPTC or XMP) in settings.", parent=self)
            return

        # Initiate the save thread for the single file
        self._initiate_save_thread([img_file], settings)

    def _get_current_save_settings(self) -> Dict[str, Any]:
        """Reads save-related settings from the app_settings dictionary."""
        return {
            "write_iptc": self.app_settings.get("write_iptc", "off") == "on" and iptc_available,
            "write_xmp": self.app_settings.get("write_xmp", "off") == "on" and xmp_available,
            "append_keywords": self.app_settings.get("append_keywords", "off") == "on"
        }

    def _initiate_save_thread(self, files_to_save: List[ImageFile], settings: Dict[str, Any]):
        """Common logic to start the metadata saving thread."""
        log_message(f"Starting metadata saving for {len(files_to_save)} files...")
        self.update_status(f"Saving metadata ({len(files_to_save)})...")
        self.progress_bar.set(0)
        self.stop_event.clear()
        for f in files_to_save:
            f.update_status(STATUS_SAVING)  # Update status in ImageFile object
        self.update_action_buttons_state()  # Disable buttons, enable Stop

        self.save_thread = threading.Thread(
            target=save_metadata_thread,
            args=(files_to_save, settings, self.progress_queue, self.stop_event),
            name="MetadataSaver",
            daemon=True
        )
        self.save_thread.start()

    def stop_current_process(self):
        """Signals the currently active worker thread to stop."""
        active_thread = None
        name = ""
        # Check which thread is currently alive
        if self.file_load_thread and self.file_load_thread.is_alive():
            active_thread = self.file_load_thread
            name = "File/Metadata Loading"
        elif self.thumb_load_thread and self.thumb_load_thread.is_alive():
            active_thread = self.thumb_load_thread
            name = "Thumbnail Loading"
        elif self.ai_process_thread and self.ai_process_thread.is_alive():
            active_thread = self.ai_process_thread
            name = "AI Processing"
        elif self.save_thread and self.save_thread.is_alive():
            active_thread = self.save_thread
            name = "Metadata Saving"

        if active_thread:
            log_message(f"Requesting stop for process: {name}...", logging.INFO)
            self.stop_event.set()  # Signal the thread to stop
            # Disable the stop button immediately and change text
            self.safe_widget_update(self.stop_button.configure, state="disabled", text="Stopping...")
            # The thread itself should handle cleanup and report stopped status via queue
        else:
            log_message("No active process to stop.", logging.INFO)
            # Ensure stop button is disabled if no process is running
            self.safe_widget_update(self.stop_button.configure, state="disabled", text="Stop Process")

    # --- State & UI Update Helpers ---
    def is_busy(self, thread_name: Optional[str] = None) -> bool:
        """Checks if a specific thread or any worker thread is currently active."""
        threads = {
            "FileMetaLoader": self.file_load_thread,
            "ThumbnailLoader": self.thumb_load_thread,
            "AIProcessor": self.ai_process_thread,
            "MetadataSaver": self.save_thread
        }
        if thread_name:
            t = threads.get(thread_name)
            return bool(t and t.is_alive())
        else:
            # Check if any of the tracked threads are alive
            return any(t and t.is_alive() for t in threads.values())

    def update_action_buttons_state(self):
        """Updates the enabled/disabled state of all major action buttons."""
        # Schedule this to run on the main thread to avoid Tkinter errors
        self.after(0, self._update_action_buttons_state_sync)

    def _update_action_buttons_state_sync(self):
        """Synchronous part of updating button states (runs in main thread)."""
        if not self.winfo_exists():
            return  # Exit if window is destroyed

        # --- Determine current state ---
        busy = self.is_busy()
        has_files = bool(self.image_files)
        selected_files = self.get_selected_files()
        has_selection = bool(selected_files)
        has_dirty_selection = any(f.is_dirty for f in selected_files)

        can_save_current = False
        can_edit_detail = False
        if self.currently_selected_file and self.currently_selected_file in self.image_files:
            current_file = self.image_files[self.currently_selected_file]
            can_save_current = current_file.is_dirty and not busy  # Can save current if dirty and not busy
            # Can edit detail if processed/ready and AI is not running
            can_edit_detail = current_file.status in [
                STATUS_PROCESSED_OK, STATUS_SAVED, STATUS_SAVE_ERROR,
                STATUS_READY, STATUS_PROCESSED_ERR, STATUS_THUMB_ERROR, STATUS_LOAD_ERROR
            ] and not self.is_busy("AIProcessor") and not self.is_busy("FileMetaLoader")

        # --- Update Top Bar & List Panel Controls ---
        self.safe_widget_update(self.select_folder_button.configure, state="disabled" if busy else "normal")
        can_reload = bool(self.selected_folder and os.path.isdir(self.selected_folder))
        self.safe_widget_update(self.load_files_button.configure, state="disabled" if busy or not can_reload else "normal")
        # Update menu item state for reload
        menu_state = "normal" if can_reload and not busy else "disabled"
        if hasattr(self, 'menu_bar') and hasattr(self, 'reload_menu_item_index'):
            try:
                self.menu_bar.entryconfigure(self.reload_menu_item_index, state=menu_state)
            except tk.TclError:
                pass  # Ignore if menu doesn't exist yet

        # Settings menu item - disable if busy? Generally okay to open.
        # if hasattr(self, 'menu_bar'): ... entryconfigure ...

        # List panel select buttons
        self.safe_widget_update(self.list_panel.select_all_button.configure, state="disabled" if busy or not has_files else "normal")
        self.safe_widget_update(self.list_panel.select_none_button.configure, state="disabled" if busy or not has_files else "normal")

        # --- Update Status Bar Controls ---
        self.safe_widget_update(self.process_button.configure, state="disabled" if busy or not has_selection else "normal")
        self.safe_widget_update(self.save_selected_button.configure, state="disabled" if busy or not has_dirty_selection else "normal")
        self.safe_widget_update(self.stop_button.configure, state="normal" if busy else "disabled")
        # Reset stop button text if not busy
        if not busy:
            self.safe_widget_update(self.stop_button.configure, text="Stop Process")

        # --- Update Detail Panel Controls ---
        self.safe_widget_update(self.detail_panel.save_current_button.configure, state="disabled" if busy or not can_save_current else "normal")
        # Enable/disable editable text fields
        edit_state = "normal" if can_edit_detail else "disabled"
        self.safe_widget_update(self.detail_panel.generated_caption.configure, state=edit_state)
        self.safe_widget_update(self.detail_panel.generated_keywords.configure, state=edit_state)

    def update_status(self, text: str, level: str = "info"):
        """Updates the main status label text and color, and logs the message."""
        colors = {"info": "white", "warning": "orange", "error": "red", "success": "lightgreen", "debug": "gray"}
        color = colors.get(level, "white")
        # Schedule GUI update on main thread
        self.after(0, lambda t=text, c=color: self.safe_widget_update(self.status_label.configure, text=t, text_color=c))
        # Log the message using the appropriate level
        log_level_map = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR, "success": logging.INFO}
        logger.log(log_level_map.get(level, logging.INFO), f"STATUS: {text}")

    # --- Queue Processing ---
    def check_queue(self):
        """Periodically checks the queue for messages from worker threads."""
        try:
            while True:  # Process all messages currently in queue
                message = self.progress_queue.get_nowait()
                msg_type = message.get("type")
                data = message.get("data", {})
                logger.debug(f"Queue received: {msg_type}")

                # --- File/Metadata Loading ---
                if msg_type == "file_found_pending":
                    path, base = data.get("path"), data.get("base_name")
                    if path and base and path not in self.image_files:
                        img_file = ImageFile(path)
                        self.image_files[path] = img_file
                        # Delegate adding UI item to ListPanel
                        if hasattr(self, 'list_panel') and self.list_panel:
                            self.list_panel.add_item(img_file)
                            # No need to update count here, add_item handles it
                elif msg_type == "file_scan_found_count":
                    self.update_status(f"Found {data.get('count', 0)} images. Loading metadata...")
                elif msg_type == "metadata_load_progress":
                    self.safe_widget_update(self.progress_bar.set, data.get("progress", 0))
                elif msg_type == "metadata_load_done":
                    files_data, count = data.get("files_data", []), data.get("count", 0)
                    self.update_status(f"Metadata loaded for {count}. Loading thumbnails...")
                    self.safe_widget_update(self.progress_bar.set, 0)  # Reset progress for thumbnails
                    loaded_files_for_thumbs = []
                    for info in files_data:
                        path = info.get("path")
                        if path and path in self.image_files:
                            img = self.image_files[path]
                            img.set_existing_meta(info.get("existing_caption"), info.get("existing_keywords", []), info.get("existing_source"))
                            loaded_files_for_thumbs.append(img)
                        else:
                            logger.warning(f"Metadata received for unknown or missing file: {path}")
                    # Mark file loading thread as finished
                    self.file_load_thread = None
                    self.update_action_buttons_state()  # Update buttons now that file load is done
                    # Start thumbnail loading
                    self.start_load_thumbnails(loaded_files_for_thumbs)
                elif msg_type == "file_scan_stopped" or msg_type == "metadata_load_stopped":
                    self.update_status("Metadata loading stopped.", "warning")
                    self.file_load_thread = None
                    self.update_action_buttons_state()

                # --- Thumbnail Loading ---
                elif msg_type == "thumbnail_loaded":
                    path = data.get("path")
                    list_widget = self.list_panel.list_item_widgets.get(path)
                    if list_widget:
                        list_widget.update_thumbnail(data.get("thumbnail"))
                elif msg_type == "thumbnail_error":
                    path = data.get("path")
                    list_widget = self.list_panel.list_item_widgets.get(path)
                    if list_widget:
                        list_widget.update_thumbnail(error=True)
                elif msg_type == "thumbnail_load_done":
                    self.update_status("Thumbnails loaded.", "info")
                    self.thumb_load_thread = None
                    self.update_action_buttons_state()

                # --- AI Processing & Saving ---
                elif msg_type == "update_status_list":
                    path, status, details = data.get("path"), data.get("status"), data.get("details", "")
                    if path in self.image_files:
                        img_file = self.image_files[path]
                        img_file.update_status(status, details)  # This triggers list item update
                        # If this is the currently selected file, refresh detail panel
                        if self.currently_selected_file == path:
                            self.display_details(img_file)
                    else:
                        logger.warning(f"Status update for unknown file: {path}")
                elif msg_type == "status":
                    self.update_status(data.get("text", "..."), data.get("level", "info"))
                elif msg_type == "ai_progress" or msg_type == "save_progress":
                    self.safe_widget_update(self.progress_bar.set, data.get("progress", 0))
                elif msg_type == "ai_result":
                    path, cap, kw = data.get("path"), data.get("caption"), data.get("keywords")
                    if path in self.image_files:
                        img_file = self.image_files[path]
                        img_file.set_generated(cap, kw)  # This updates status and dirty flag
                        # If this is the currently selected file, refresh detail panel
                        if self.currently_selected_file == path:
                            self.display_details(img_file)
                        # Update buttons as file might now be dirty/processed
                        self.update_action_buttons_state()
                    else:
                        logger.warning(f"AI result for unknown file: {path}")
                elif msg_type == "ai_done":
                    proc, total, stopped, err = data.get("processed", 0), data.get("total", 0), data.get("stopped", False), data.get("error", False)
                    # Set progress to 1 only if finished successfully
                    if not stopped and not err and total > 0:
                        self.safe_widget_update(self.progress_bar.set, 1.0)
                    self.ai_process_thread = None  # Mark thread as finished
                    if not err:  # Don't clear stop event if there was a model load error
                        self.stop_event.clear()
                        self.safe_widget_update(self.stop_button.configure, text="Stop Process")
                    self.update_action_buttons_state()  # Re-enable buttons
                elif msg_type == "save_result":
                    path, success, err_msg = data.get("path"), data.get("success"), data.get("error_msg", "")
                    if path in self.image_files:
                        img_file = self.image_files[path]
                        if success:
                            img_file.mark_saved()  # Updates status, clears dirty flag
                        else:
                            img_file.mark_save_error(err_msg)  # Updates status, keeps dirty flag
                        # If this is the currently selected file, refresh detail panel
                        if self.currently_selected_file == path:
                            self.display_details(img_file)
                        # Update buttons (Save Selected might become disabled if all saved)
                        self.update_action_buttons_state()
                    else:
                        logger.warning(f"Save result for unknown file: {path}")
                elif msg_type == "save_done":
                    saved, total, stopped = data.get("saved", 0), data.get("total", 0), data.get("stopped", False)
                    errors = total - saved
                    # Set progress to 1 only if finished successfully without errors
                    if not stopped and errors == 0 and total > 0:
                        self.safe_widget_update(self.progress_bar.set, 1.0)
                    self.save_thread = None  # Mark thread as finished
                    self.stop_event.clear()
                    self.safe_widget_update(self.stop_button.configure, text="Stop Process")
                    self.update_action_buttons_state()  # Re-enable buttons
                elif msg_type == "error":  # Generic thread error
                    msg, t_name = data.get("message", "Unknown error"), data.get("thread_name", "Worker")
                    self.update_status(f"ERROR ({t_name}): {msg}", "error")
                    # Mark responsible thread as finished
                    if t_name == "FileMetaLoader":
                        self.file_load_thread = None
                    elif t_name == "ThumbnailLoader":
                        self.thumb_load_thread = None
                    elif t_name == "AIProcessor":
                        self.ai_process_thread = None
                    elif t_name == "MetadataSaver":
                        self.save_thread = None
                    # Update buttons as the process has failed/stopped
                    self.update_action_buttons_state()

        except Empty:
            pass  # Queue is empty, normal operation
        except Exception as e:
            logger.error(f"Unexpected error processing queue: {e}", exc_info=True)
        finally:
            # Reschedule the check only if the window still exists
            if self.winfo_exists():
                self.after(100, self.check_queue)

    def safe_widget_update(self, widget_method: Callable, *args, **kwargs):
        """Safely calls a widget method, checking if the widget exists."""
        try:
            # Get the widget instance from the bound method
            widget = widget_method.__self__
            # Check if the widget instance exists
            if widget.winfo_exists():
                widget_method(*args, **kwargs)
            # else: logger.debug(f"Widget update skipped, widget destroyed: {widget}")
        except tk.TclError as e:
            # Catch specific Tcl errors like 'invalid command name' which indicate destruction
            if "invalid command name" not in str(e):
                logger.warning(f"Safe update TclError for {widget_method.__name__}: {e}")
        except AttributeError:
            # Catch if __self__ is not available (e.g., not a bound method)
            logger.warning(f"Safe update failed: {widget_method} není vázaná metoda?")
        except Exception as e:
            # Catch any other unexpected errors during the update
            logger.error(f"Safe update error for {widget_method.__name__}: {e}", exc_info=True)

    # --- Logging & Closing ---
    def log_message(self, message: str, level=logging.INFO):
        """Adds a message to the log textbox in the ListPanel."""
        # Delegate logging to ListPanel's textbox if it exists
        if hasattr(self, 'list_panel') and self.list_panel and hasattr(self.list_panel, 'log_textbox'):
            log_textbox = self.list_panel.log_textbox
            if not log_textbox.winfo_exists():
                return  # Exit if textbox destroyed

            log_prefix_map = {
                logging.CRITICAL: "CRITICAL: ", logging.ERROR: "ERROR: ",
                logging.WARNING: "WARNING: ", logging.INFO: "", logging.DEBUG: "DEBUG: "
            }
            log_prefix = log_prefix_map.get(level, "")
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            full_message = f"{timestamp} {log_prefix}{message}\n"

            # Function to update GUI, scheduled with after(0, ...)
            def update_gui():
                try:
                    if log_textbox.winfo_exists():
                        current_state = log_textbox.cget("state")
                        log_textbox.configure(state="normal")
                        log_textbox.insert(tk.END, full_message)
                        log_textbox.configure(state=current_state)  # Restore original state
                        log_textbox.see(tk.END)  # Scroll to the end
                except tk.TclError:
                    pass  # Ignore errors if widget is destroyed between check and update
                except Exception as e:
                    print(f"GUI Log Error: {e}")  # Fallback print

            # Schedule the update on the main thread
            if self.winfo_exists():
                self.after(0, update_gui)
        else:
            # Fallback if list_panel or log_textbox doesn't exist yet/anymore
            print(f"Fallback Log: {message}")

    def on_closing(self):
        """Handles application closing sequence."""
        log_message("Closing application...", logging.INFO)
        # 1. Signal any running threads to stop
        if self.is_busy():
            log_message("Signaling stop to running threads...", logging.INFO)
            self.stop_event.set()
            # Optionally wait a very short time? Might hang GUI.
            # time.sleep(0.1)

        # 2. Save any pending edits from detail panel
        try:
            if self.currently_selected_file:
                self.update_image_data_from_detail(self.currently_selected_file, force_update=True)
        except Exception as e:
            log_message(f"Error saving detail on close: {e}", logging.WARNING)

        # 3. Save application settings (window size, last folder, etc.)
        self.save_settings()

        # 4. Destroy the main window
        log_message("Destroying main window.", logging.INFO)
        self.destroy()

# --- Global Logging Function ---
def log_message(message: str, level=logging.INFO, exc_info=False):
    """Logs message to file/console and attempts to log to GUI."""
    # Log to configured handlers (file, stream)
    logger.log(level, message, exc_info=exc_info)
    # Attempt to log to GUI if app instance exists
    global app
    try:
        # Check if app exists, is the correct type, and the window exists
        if 'app' in globals() and app is not None and isinstance(app, ImageTaggerApp) and app.winfo_exists():
            # Call the app's method to handle GUI logging
            app.log_message(message, level)
    except Exception as gui_log_e:
        # Fallback if GUI logging fails for any reason
        print(f"Fallback print (GUI log failed): {message} (Error: {gui_log_e})")

# --- Startup ---
if __name__ == "__main__":
    # Set DPI awareness for Windows (optional, improves scaling)
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
        logger.info("DPI Awareness set (Windows).")
    except Exception:
        logger.debug("Failed to set DPI awareness (probably not Windows or error).")

    try:
        # Create and run the application
        app = ImageTaggerApp()
        if app:
            app.mainloop()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        log_message("\nApplication terminated by user (Ctrl+C).", logging.INFO)
    except Exception as e:
        # Log critical errors that prevent the app from running
        log_message(f"CRITICAL APPLICATION ERROR: {e}", logging.CRITICAL, exc_info=True)
        try:
            # Attempt to show an error messagebox if possible
            messagebox.showerror("Critical Application Error", f"An unexpected critical error occurred:\n\n{e}\n\nCheck the log file: {LOG_FILENAME}")
        except Exception as mb_e:
            # Fallback if even the messagebox fails
            print(f"FATAL: Cannot show error messagebox: {mb_e}")
    finally:
        # Final log message before exit
        log_message("Application terminated.", logging.INFO)
        # Explicitly close file handler to ensure logs are flushed
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
