# Hugging Face Space Setup Guide

To deploy this Love Live! Wordle AI, follow these settings when creating your new Space:

## 1. Create New Space
*   **Space Name:** (Choose anything, e.g., `lovelive-wordle-ai`)
*   **License:** MIT (or your preference)
*   **Select the Space SDK:** **Gradio**
*   **Space Hardware:** **CPU basic (free)**
    *   *Note:* The model is very lightweight (< 1MB), so the free tier is perfectly adequate.
*   **Visibility:** Public

## 2. Upload Files
Once the space is created, upload the following files from this repository to the "Files" tab of your Space:

1.  `app.py`
2.  `requirements.txt`
3.  `game.py`
4.  `model.py`
5.  `game_data.json`
6.  `mappings.json`
7.  `transformer_model.pth`

## 3. Deployment
Hugging Face will automatically install the dependencies listed in `requirements.txt` and launch `app.py`. The "Building" status should change to "Running" within a minute or two.
