# MidiGPT

A transformer-based MIDI generation system that encodes MIDI files into tokens and uses a GPT-2 style model to learn and generate musical sequences.

## Features

- **MIDI Encoding**: Converts MIDI files into discrete tokens representing musical events
  - Sustain pedal processing (extends note durations based on CC64 events)
  - Velocity quantization (32 bins by default to reduce vocabulary size)
  - Tempo change tracking
  - Time-based event ordering with millisecond precision

- **MIDI Decoding**: Reconstructs MIDI files from token sequences

- **Transformer Model**: GPT-2 style architecture for music generation
  - Multi-head causal self-attention
  - Position embeddings
  - Layer normalization
  - Feed-forward networks
  - Weight tying between token embeddings and output layer

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- `pretty_midi`: MIDI file processing
- `numpy`: Numerical operations
- `mido`: MIDI I/O operations
- `torch`: PyTorch for the transformer model

## Usage

### Basic MIDI Encoding/Decoding

Run the main script to encode a MIDI file and decode it back:

```bash
python main.py
```

This will:
1. Encode `data/sample1.mid` to tokens
2. Save tokens to `data/sample1.tokens.json`
3. Decode tokens back to `data/output.mid`

### Training the Transformer Model

Run the transformer script to train on MIDI data and generate new music:

```bash
python transformer.py
```

This will:
1. Encode a MIDI file to tokens
2. Train a GPT-2 style model on the token sequence
3. Generate new musical sequences
4. Save the generated output to `data/output_tranformer.mid`

### Custom Usage

```python
from encode import encode_midi
from decode import decode_tokens

# Encode MIDI to tokens
tokens = encode_midi('path/to/input.mid')

# Decode tokens back to MIDI
decode_tokens(tokens, 'path/to/output.mid')
```

## Project Structure

```
MidiGPT/
├── encode.py           # MIDI to token encoding with sustain pedal support
├── decode.py           # Token to MIDI decoding
├── transformer.py      # GPT-2 model training and generation
├── main.py            # CLI entry point for encoding/decoding
├── requirements.txt   # Python dependencies
└── data/              # MIDI files and outputs
    ├── sample1.mid
    ├── sample2.mid
    └── output*.mid
```

## How It Works

### Token Encoding

The encoding process transforms MIDI into a sequence of discrete tokens:

1. **Sustain Pedal Processing**: Extends note durations based on sustain pedal events (CC64)
2. **Event Collection**: Gathers note ON/OFF events and tempo changes
3. **Event Sorting**: Orders events by time, then by priority (OFF → TEMPO → ON)
4. **Tokenization**: Converts events to tokens:
   - `TIME_SHIFT_{ms}`: Time advance in milliseconds
   - `NOTE_ON_{pitch}_{velocity_bin}`: Note start with pitch and quantized velocity
   - `NOTE_OFF_{pitch}`: Note end
   - `TEMPO_{bpm}`: Tempo change (rounded to nearest 4 BPM)

### Token Decoding

Reconstructs MIDI from tokens by:
1. Tracking current time and active notes
2. Processing time shifts to advance playback position
3. Starting/stopping notes based on ON/OFF tokens
4. Applying tempo changes

### Model Architecture

The GPT-2 style transformer includes:
- **Embedding Layer**: Token and positional embeddings (64 dimensions)
- **Transformer Blocks** (4 layers):
  - Multi-head causal self-attention (4 heads)
  - Feed-forward network with GELU activation
  - Layer normalization (pre-norm architecture)
- **Output Head**: Linear projection to vocabulary with weight tying

Training uses cross-entropy loss with AdamW optimizer, learning musical patterns from the tokenized MIDI sequences.

## Model Configuration

Default hyperparameters in `transformer.py`:
- Sequence length: 5 tokens
- Model dimension: 64
- Number of layers: 4
- Number of attention heads: 4
- Training epochs: 400
- Learning rate: 1e-3

## Device Support

The transformer model automatically detects and uses:
- Apple Silicon GPU (MPS) if available
- CPU as fallback

## License

This project is provided as-is for educational and research purposes.