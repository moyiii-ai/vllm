## Manual Setup (For Multi-GPU Binding)

If multiple GPUs need to be bound to the container, use this manual method to avoid Dev Container conflicts:

1. Clone the repository: `git clone https://github.com/moyiii-ai/vllm.git`
2. Run the setup script directly: `./vllm/srs_setup/start_vllm_container.sh`
3. Connect to the running container:
    * Either via Docker exec: docker exec -it <container_name> bash
    * Or through VS Code: Use "Dev Containers: Attach to Running Container" and select the container
4. Initialize the environment inside the container: `/vllm-workspace/vllm/srs_setup/container_init.sh`

## Automatic Setup (No GPU Binding)

For environments without GPU binding requirements, use VS Code's automatic Dev Container setup. Control GPU usage within the container as needed.

1. Prepare the workspace, for example: `mkdir my-workspace && cd my-workspace`
2. Clone the repository: `git clone https://github.com/moyiii-ai/vllm.git`
3. Copy the .devcontainer folder to workspace root: `cp -rv vllm/srs_setup/.devcontainer .`
4. Open my-workspace in VS Code. Use "Dev Containers: Rebuild and Reopen in Container" to initialize.
    * Subsequent launches will connect automatically.
    * Rebuild only when changes are made to devcontainer.json.


## Steps to run vLLM

1. Set Up the Hugging Face Token: `echo 'export HUGGING_FACE_HUB_TOKEN=hf_xxx' >> ~/.bashrc && source ~/.bashrc`. 

Note: If you plan to use Llama-series models, you must first apply for access to the model on Hugging Face. After approval, update your token’s permissions: Go to Settings > Access Tokens > edit access token permissions, then in the Repository permissions section, explicitly add the Llama model repository.
2. Navigate to the vLLM test directory: `cd vllm/srs_test`
3. Run the relevant scripts or commands