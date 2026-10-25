"""Populate and verify the Hugging Face cache needed by SimKGC."""
import argparse
import os
import time

os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '600')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '60')

from transformers import AutoModel, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bert-base-uncased')
    parser.add_argument('--attempts', type=int, default=20)
    parser.add_argument('--delay', type=int, default=10)
    args = parser.parse_args()

    for attempt in range(1, args.attempts + 1):
        try:
            print('SimKGC model-cache attempt {}/{}: {}'.format(
                attempt, args.attempts, args.model), flush=True)
            AutoTokenizer.from_pretrained(args.model)
            AutoModel.from_pretrained(args.model)
            print('SimKGC pretrained model cache verified', flush=True)
            return
        except Exception as error:
            print('Download/cache verification failed: {!r}'.format(error), flush=True)
            if attempt == args.attempts:
                raise
            time.sleep(args.delay)


if __name__ == '__main__':
    main()
