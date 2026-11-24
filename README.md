# PyTorch Lesson 15: GPT Text Training

**Author:** lelexiaomz

A small educational example that demonstrates how a GPT-style Transformer can learn next-token prediction from a short English text. The project includes text preprocessing, vocabulary creation, sliding-window datasets, multi-head causal self-attention, training, validation, testing, early stopping, and next-token prediction.

> This is a learning project, not a production language model. The corpus is small, so the model can overfit and its predictions are limited to patterns and tokens in this text.

## Files

- `pytorch lesson 15_gpt_text_training.py` — model, data preparation, training, evaluation, and prediction code.
- `lesson_15_training_text.txt` — English text corpus used by the program.

Keep these two files in the same directory. The script locates the text file relative to its own location.

## Requirements

- Python 3
- PyTorch
- Matplotlib

Install the dependencies in a terminal:

```powershell
py -m pip install torch matplotlib
```

If `py` is not available on your system, use your Python environment's pip command instead.

## Run

From the directory containing both files, run:

```powershell
py "pytorch lesson 15_gpt_text_training.py"
```

The script prints dataset and training information, evaluates the model, displays training/validation loss curves, and predicts a token after the example prompt configured near the end of the script.

## How the example works

1. The tokenizer lowercases the text, treats runs of word characters as tokens, and separates punctuation.
2. A vocabulary maps each token to an integer ID. The vocabulary is built from the full text file.
3. The token sequence is split in order into training, validation, and test portions (80% / 10% / 10%). Sliding windows create input and one-token-shifted target sequences; windows are created separately for each split.
4. A causal Transformer predicts the next token at each position. The example uses sequence length 4, `d_model=32`, `d_ff=64`, 4 attention heads, and 2 Transformer blocks.
5. The model is trained with cross-entropy loss and Adam. Early stopping uses a patience of 10 epochs without validation-loss improvement.
6. The prediction prompt must contain only tokens that occur in the corpus. An unknown token currently raises `KeyError` because the example does not implement an unknown-token (`<UNK>`) mechanism.

## Important learning notes

- The vocabulary is built from the entire text before the token stream is split. This is convenient for a small tutorial, but it exposes validation/test vocabulary to preprocessing. For rigorous experiments, build the vocabulary from training data only and define how unseen validation/test tokens are handled.
- The dataset is intentionally tiny. A low training loss together with a rising validation loss is a sign of overfitting; consider more text, a smaller model, fewer training epochs, or other regularization.
- Early stopping halts training after the configured patience. To save a best-validation checkpoint, pass a path through `model_save_path` when calling `train_model`; the current main call does not provide one or reload a saved best checkpoint.

## License

No license is specified yet. Add a license (for example, MIT) only if you have the right to grant others permission to use, modify, and redistribute all included code and text.



