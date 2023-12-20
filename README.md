# tabbyAPI-gradio-loader
A simple Gradio WebUI for loading/unloading models and loras for tabbyAPI. This provides demo functionality for accessing tabbyAPI's extensive feature base via API, and can be run remotely on a separate system.

## Usage
This repo is meant serve as a demo of the API's features and provide an accessible means to change models without editing the config and restarting the instance. Supports speculative decoding and loading of multiple loras with custom scaling.

This WebUI does not provide an LLM inference frontend - use any OAI-compatible inference frontend of your choosing.

## Prerequisites

To get started, make sure you have the following installed on your system:

- Python 3.8+ (preferably 3.11) with pip

## Installation

1. Clone this repository to your machine: `git clone https://github.com/theroyallab/tabbyAPI-gradio-loader`
2. Navigate to the project directory: `cd tabbyAPI-gradio-loader`
3. Create a python virtual environment: `python -m venv venv`
4. Activate the virtual environment:
   1. On Windows (Using powershell or Windows terminal): `.\venv\Scripts\activate.`
   2. On Linux: `source venv/bin/activate`
5. Install the requirements file: `pip install -r requirements.txt`

## Launching the Application
1. Make sure you are in the project directory and entered into the venv
2. Run the WebUI application: `python webui.py`
3. Input your tabbyAPI endpoint URL and admin key and press connect!

## Command-line Arguments
| Argument           | Description                                               |
| :----------------  | :-----------------------------------------------------    |
| `-h` or`--help`    |  Show this help message and exit                          |
| `-p` or `--port`   |  Specify port to host the WebUI on (default 7860)         |
| `-l` or `--listen` |  Share WebUI link via LAN                                 |
| `-s` or `--share`  |  Share WebUI link remotely via Gradio's built in tunnel   |
