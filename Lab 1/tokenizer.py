import re

def tokenizer(sentence):
    URL_PATTERN = r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
    EMAIL_PATTERN = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?'
    DATE_PATTERN = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
    DECIMAL_PATTERN = r'\d+\.\d+'
    NUMBER_PATTERN = r'\d+'
    PUNCT_PATTERN = r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*|[^\w\s]"

    token_re = re.compile(
        f"{URL_PATTERN}|{EMAIL_PATTERN}|{DATE_PATTERN}|{DECIMAL_PATTERN}|{NUMBER_PATTERN}|{PUNCT_PATTERN}"
    )

    return token_re.findall(sentence)


def word_tokenizer(sentences):
    tokenized_words = []
    for sent in sentences:
        tokens = tokenizer(sent)
        for token in tokens:
            if len(token) < 3:
                continue
            else:
                tokenized_words.append(token)
    
    return tokenized_words

def tokenize_sentence(paragraph: str):
    #protect urls, dates, emails
    URL_PATTERN = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
    EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?$'
    DATE_PATTERN   = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'

    protected = []
    def protect(match):
        protected.append(match.group(0))
        return f"__PROTECTED__{len(protected)-1}__"
    
    paragraph = re.sub(r'\.\.\.', protect, paragraph)
    paragraph = re.sub(URL_PATTERN, protect, paragraph)
    paragraph = re.sub(EMAIL_PATTERN, protect, paragraph)
    paragraph = re.sub(DATE_PATTERN, protect, paragraph)

    text = paragraph.strip().replace("\n", " ")

    sentence_end = re.compile(r"([\.?!])\s+")

    parts = sentence_end.split(text)

    sentences = []
    for i in range(0, len(parts)-1, 2):
        if i+1 < len(parts):
            sentence = parts[i] + parts[i+1]
        else:
            sentence = parts[i]

        for idx, item in enumerate(protected):
            sentence = sentence.replace(f"__PROTECTED__{idx}__", item)
        
        sentence = sentence.strip()
        if sentence:
            sentences.append(sentence)
        
    
    if len(parts) % 2 == 1 and parts[-1].strip():
        last = parts[-1].strip()

        for idx, item in enumerate(protected):
            last = last.replace(f"__PROTECTED__{idx}__", item)
        
        if last:
            sentences.append(last)
    
    return sentences
