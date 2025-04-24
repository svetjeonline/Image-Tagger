# Image-Tagger
# AI Image Tagger & Describer

## Overview

The AI Image Tagger & Describer is a desktop application designed to help you automatically generate English captions and keywords for your images using AI, and write this metadata to IPTC/XMP.  This is useful for organizing, cataloging, and improving the searchability of your image libraries.

## Features

* **AI-Powered Captioning and Keyword Generation:** Uses the Salesforce BLIP model (Base or Large) via Hugging Face Transformers.
* **Metadata Writing:** Writes generated (and edited) captions and keywords to IPTC and XMP metadata within the image files.
* **Cross-Platform Compatibility:** Built with Python and Tkinter, should run on Windows, macOS, and Linux.
* **User-Friendly Interface:**
    * Image List Panel with status indicators and selection checkboxes.
    * Detail Panel with large preview and editable metadata fields.
* **Settings Dialog:** Customize AI model, processing device (CPU/GPU), metadata writing options, and stop words.
* **Batch Processing:** Generate and save metadata for multiple images at once.
* **Logging:** Detailed logging to both a file and the GUI log textbox.
* **Error Handling:** Robust error handling with informative message boxes.
* **Virtual Environment:** Uses a virtual environment to manage dependencies.

## Key Technologies

* Python 3.9+
* CustomTkinter
* Hugging Face Transformers (BLIP)
* PyTorch (optional, for GPU acceleration)
* iptcinfo3 (for IPTC metadata)
* python-xmp-toolkit (for XMP metadata)
* libexempi (system library for XMP)

## Application Interface
The application is divided into two main panels:

### List Panel
* Displays a list of images from the selected folder
* Shows a small thumbnail preview of each image
* Displays the name of each image
* Shows processing status for each image
* Checkbox to select images for processing

### Detail Panel
* Displays a larger preview of the selected image
* Shows existing metadata
* Shows AI generated and editable metadata
* Save button to save the metadata of the selected image

## How it Works

1.  **Image Loading:** The application scans a selected folder for supported image files (JPG, JPEG, TIFF).
2.  **Metadata Reading:** Existing IPTC and XMP metadata is read from the image files.
3.  **AI Processing:**
    * The BLIP model generates an English caption and keywords for each selected image.
4.  **Metadata Writing:**
    * The generated (or edited) metadata is written back to the image files as IPTC and/or XMP metadata, according to user settings.

## Setup Instructions

### Prerequisites

* **Python:** Python 3.9 or later is required.  Make sure  "Add Python to PATH" is checked during installation.
* **pip:** Python's package installer.  Usually included with Python.
* **C++ Build Tools:** If you encounter errors during the installation of python-xmp-toolkit, you might need to install Microsoft Visual C++ Build Tools.
* **Exempi:** The Exempi library is required for XMP metadata handling.  The application attempts to install this automatically on Windows using Conda, but manual installation may be required.

### Installation Steps

1.  **Download the repository:** Clone the repository to your local machine.
2.  **Create a virtual environment:** Open a command prompt or terminal, navigate to the project directory, and run:
    ```bash
    python -m venv .venv
    ```
3.  **Activate the virtual environment:**
    * On Windows, run:
        ```bash
        .venv\Scripts\activate
        ```
4.  **Install dependencies:** In the activated virtual environment, run:
        ```bash
        pip install -r requirements.txt
        ```
5.  **Install Exempi (Windows):**
    * The application will try to install Exempi automatically using Conda. If this fails, follow the manual installation instructions provided by the application.
    * Manual installation involves downloading a pre-built binary, extracting it, and adding the path to the system PATH environment variable.
6.  **Run the application:** In the activated virtual environment, run:
    ```bash
    python main.py
    ```

##  Screenshots
[Include screenshots of the application's main window, settings dialog, and about dialog]

##  License
[Specify the license under which the project is released]

##  Contributions
[State if you accept contributions and how to contribute]

##  Credits
[Acknowledge any libraries, frameworks, or resources used]
