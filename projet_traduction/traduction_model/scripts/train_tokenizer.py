import sentencepiece as spm

def train_tokenizer(file_path, vocab_size, name):
    spm.SentencePieceTrainer.Train(
        input=file_path,
        model_prefix=name,
        vocab_size=vocab_size,
        character_coverage=1.0,
        user_defined_symbols=["<s>","<pad>", "</s>"],
        model_type='bpe'
    )

def load_tokenizer(model_name):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(model_name)
    return tokenizer
