Steps to setup:

1. Create the workspace folder, for example: `mkdir my-workspace`
2. Navigate into it: `cd my-workspace`
3. Clone the repository: `git clone https://github.com/moyiii-ai/vllm.git`
4. Copy the .devcontainer folder to workspace root: `cp -rv vllm/srs_setup/.devcontainer .`
5. Open the my-workspace folder in VS Code, then rebuild and reopen it using the dev container. For subsequent connections, the dev container will open automatically—rebuilding is only necessary when changes are made to devcontainer.json.


Steps to run vLLM:

1. Set Up the Hugging Face Token: `echo 'export HUGGING_FACE_HUB_TOKEN=hf_xxx' >> ~/.bashrc && source ~/.bashrc`. 
Note: If you plan to use Llama-series models, you must first apply for access to the model on Hugging Face. After approval, update your token’s permissions: Go to Settings > Access Tokens > edit access token permissions, then in the Repository permissions section, explicitly add the Llama model repository.
2. Navigate to the vLLM test directory: `cd vllm/srs_test`
3. Run the relevant scripts or commands