# CoTR_Prompting_low_resource

Chain of Translation Reasoning (CoTR) for Low-Resource Languages

## Natural Language Inference (NLI)

The NLI module compares baseline and CoTR approaches for natural language inference tasks. It supports various languages from the XNLI dataset, focusing on English, Swahili, and Hausa.

### Running NLI Experiments

Basic usage:
```
python src/experiments/run_nli.py
```

By default, this will:
- Run both baseline and CoTR approaches
- Use both Qwen and Aya models
- Test on English, Swahili, and Hausa
- Use 100 samples per language
- Use greedy decoding (temperature 0.3)

### NLI Command-Line Options

Customize your experiments with these options:

- `--langs`: Languages to test, e.g., `--langs en sw ha`
- `--model`: Model to use (`aya`, `qwen`, or `both`)
- `--approach`: Evaluation approach (`baseline`, `cotr`, or `both`)
- `--samples`: Number of samples per language
- `--temperature`: Temperature for text generation (0.1-1.0)
- `--max-tokens`: Maximum tokens to generate for NLI
- `--translation-temp`: Temperature for translation in CoTR
- `--do-sample`: Enable sampling (vs. greedy decoding)

### Example NLI Commands

Run only Swahili with 20 samples:
```
python src/experiments/run_nli.py --langs sw --samples 20
```

Use higher temperature with sampling for more diverse outputs:
```
python src/experiments/run_nli.py --temperature 0.7 --do-sample
```

Run only CoTR with the Aya model:
```
python src/experiments/run_nli.py --model aya --approach cotr
```

Customize translation parameters:
```
python src/experiments/run_nli.py --translation-temp 0.2 --temperature 0.5 --do-sample
```

### Notes on Generation Parameters

- **Temperature**: Controls randomness in generation (0.1-1.0)
  - Lower values (0.1-0.3): More deterministic, focused
  - Higher values (0.7-1.0): More diverse outputs
  - Note: Temperature has no effect unless `--do-sample` is used!

- **Sampling**: Use `--do-sample` to enable non-greedy sampling
  - Without this flag, temperature settings have no effect
  - Required for temperature-based generation

- **Translation Temperature**: For CoTR only
  - Controls randomness in translation step
  - Can be set independently from main inference temperature