#!/usr/bin/env python3
from encode import encode_midi
import json
import sys
from pathlib import Path
from typing import Optional
from decode import decode_tokens


def main(midi_path: str = 'data/sample2.mid', out_path: Optional[str] = None):
    """Encode a MIDI file and write tokens to a JSON file.

    Args:
        midi_path: Path to the input MIDI file (default: data/sample1.mid).
        out_path: Optional output path for the token JSON. If not provided,
                  uses the same name with suffix .tokens.json.
    """
    tokens = encode_midi(midi_path)
    midi_p = Path(midi_path)
    if out_path is None:
        out_path = midi_p.with_suffix('.tokens.json')
    with open(out_path, 'w') as f:
        json.dump(tokens, f, indent=2)
    print(f"Encoded {midi_path} -> {out_path} ({len(tokens)} tokens)")

    with open(out_path, 'r') as f:
        tokens = json.load(f)

    decode_tokens(tokens, 'data/output.mid')

if __name__ == '__main__':
    # Usage: python main.py [midi_path] [out_path]
    midi = 'data/sample1.mid'
    main(midi, None)

