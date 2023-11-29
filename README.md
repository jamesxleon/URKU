# URKU
 Towards a kichwa-speaking LLM

# Toolkit: Text Generation Web UI

> (An implementation by oobabooga)[https://github.com/oobabooga/text-generation-webui]

A Gradio web UI for Large Language Models.

## Features=

- 3 interface modes: default, notebook, and chat.
- Supports multiple model backends and LoRAs (Low Rank Adaptations).
- Dropdown menu for model switching.
- Instruction templates for chat mode.
- Markdown output with LaTeX rendering.
- OpenAI-compatible API server.

## Installation

### One-click installers

1. Clone or download the repository.
2. Run the appropriate start script for your OS (`start_linux.sh`, `start_windows.bat`, `start_macos.sh`, `start_wsl.bat`).
3. Select your GPU vendor.
4. Have fun!

The script sets up a Conda environment in `installer_files`. To update, run the corresponding `update` script. For manual installations in `installer_files`, use the `cmd` script. 

### Manual Installation Using Conda

1. Install Conda.
2. Create and activate a new conda environment.
3. Install Pytorch (commands vary based on system and GPU).
4. Clone the repository and install dependencies from the requirements file that matches your setup&#8203;``【oaicite:2】``&#8203;.

### Alternative: Docker

Use the provided Docker files and follow the instructions to set up using Docker&#8203;``【oaicite:1】``&#8203;.

## Downloading Models

Place models in the `text-generation-webui/models` folder. Transformers or GPTQ models consist of several files and should be placed in a subfolder.

## Contributing

Contributions are welcome. Please refer to the repository's contributing guidelines for more details.

## Acknowledgments

Credit to all contributors and the community for support and development.
