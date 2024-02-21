from transformers import CLIPTokenizerFast
from collections import Counter
import json, struct, os

# Initialize the CLIP tokenizer
if not os.path.exists("clip_tokenizer"):
    clip_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    clip_tokenizer.save_pretrained("clip_tokenizer")
    print("Tokenizer saved to clip_tokenizer")
else:
    clip_tokenizer = CLIPTokenizerFast.from_pretrained("clip_tokenizer")
    print("Tokenizer loaded from clip_tokenizer")

# Compute scores, convert to .bin
in_file = "clip_tokenizer/tokenizer.json"
out_file = "tokenizer_clip.bin"
start_id = "<|startoftext|>"
end_id = "<|endoftext|>"

if __name__ == "__main__":
    with open(in_file, "r") as json_file:
        data = json.load(json_file)
        merges = data["model"]["merges"]
        data = data["model"]["vocab"]
        tokens = []
        scores = []
        for key in data.keys():
            processed_k = key
            if processed_k == start_id:
                processed_k = "\n<s>\n"
            elif processed_k == end_id:
                processed_k = "\n</s>\n"
            processed_k = processed_k.encode("utf-8")
            tokens.append(processed_k)

            # The token score is the frequency of the token in the "merges" dataset
            key_score = 0.0
            for merge in merges:
                key_score += merge.count(key)
            scores.append(key_score)

    # This section was taken from Karpathy's implementation of Llama2 in C: https://github.com/karpathy/llama2.c/blob/master/tokenizer.py
    max_token_length = max(len(k) for k in tokens)
    with open(out_file, "wb") as f:
        f.write(struct.pack("I", max_token_length))
        for bytes, score in zip(tokens, scores):
            f.write(struct.pack("fI", score, len(bytes)))
            f.write(bytes)

    print(f"Mappings saved to {out_file}")
