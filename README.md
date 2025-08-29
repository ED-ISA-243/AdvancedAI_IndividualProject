# Recipe Generator – Advanced AI Project

This repository contains the complete Python source code for my Advanced AI project, which focuses on fine-tuning and deploying a recipe generation model. In addition to the code, it includes supplementary files that collectively contribute to the finalized project.

### Data Files
- val.jsonl is included in the repository.
- train_lora.jsonl is not included due to its large size, although it shares the same data structure as val.jsonl. The file is simply too large to host on GitHub.

### Model Files
- The base model, Llama-3.2-1B-Instruct, and the fine-tuned LoRA adapters are not part of this repository.
- These models are several gigabytes in size and exceed GitHub’s storage limits.
- Consequently, this repository is code-only. While you can study and evaluate the scripts fully, running them requires access to my exact local model setup, which is not provided here.

### Usage and Setup
- Despite the absence of the model files due to size constraints, the repository clearly demonstrates the training pipeline, code structure, and integration within a desktop application.
- Hypothetically, if the models were included, a user could simply clone the repository containing the app, open it, and run it directly. The built-in virtual environment includes all necessary packages to facilitate this.

### Additional Information
- The included screenshot illustrates the training progress of the model on the entire dataset, highlighting the lengthy duration required for training.
- A demonstration video (RecipeGeneratorUsage.mp4) is included in the repository, showing the application in action.


### Important note
This project is my August/September retake. I chose to continue with the recipe generator because I find it interesting, but this version is very different from my earlier one. In this project, I fine-tuned a model on prepared data, so the improvement is in the project as a whole rather than in single parts. Even though the recipe generator here works less well than my earlier version, it shows how fine-tuning changes a model and leads to more interesting results, which is the main goal of the project.
