import pickle
import sys

PICKLE_PATH = r"F:\NLP Lab\Lab 7\ngram_models.pkl"
from ngram_model_def import NGramLanguageModel

def main():
    try:
        with open(PICKLE_PATH, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print('ERROR loading pickle:', e)
        sys.exit(2)

    print('Loaded object type:', type(data))
    if hasattr(data, 'keys'):
        keys = list(data.keys())
        print('Top-level keys:', keys)
    else:
        print('Top-level object is not a dict; repr:', repr(data)[:200])
        return

    for k in keys:
        v = data[k]
        try:
            l = len(v)
        except Exception:
            l = 'N/A'
        print('\nKEY:', k, 'type:', type(v), 'len:', l)
        if isinstance(v, dict):
            print('  sample entries:')
            for i, (kk, vv) in enumerate(v.items()):
                if i >= 5:
                    break
                t = type(vv)
                try:
                    lv = len(vv)
                except Exception:
                    lv = 'N/A'
                print('   ', i, 'key=', kk, '-> type=', t, 'len=', lv)
        else:
            # try to show a small sample repr
            try:
                print('  repr sample:', repr(v)[:200])
            except Exception:
                pass
            # Print NGramLanguageModel summary if applicable
            if isinstance(v, NGramLanguageModel):
                print('  NGramLanguageModel summary:')
                print('    n:', getattr(v, 'n', 'N/A'))
                ngram_counts = getattr(v, 'ngram_counts', None)
                if ngram_counts is not None:
                    print('    ngram_counts entries:', len(ngram_counts))
                else:
                    print('    ngram_counts: not found')

    print('\nINSPECTION COMPLETE')

if __name__ == '__main__':
    main()
