# Sigclip and Paligemma files


These files show how to utilize Sigclip and Paligemma:

- hf_token.py - in .gitignore will have to add
- netron_out.py - creates netron compatible onnx file for visualization
- testsiglip.py - shows simple usage for vector outputs
- siglipdemo.py - shows different categories and more in depth usage

## Steps

0. Create hf_token.py with read enabled huggingface token:

```py
hf_token="<your-hf-token-here>"
```

1. Run `testsiglip.py` for simple usage

```py
python3 testsiglip.py
```

2. Run `siglipdemo.py` for example of more complex comparisons

```py
python3 siglipdemo.py
```

3. Run `netron_out.py` for export of onnx file:
```py
python3 netron_out.py 
```


## Next Steps

- Export of PaliGemma Siglip Weights
- Dissected PaliGemma model for ingesting pre-tokenized images + text prompt

