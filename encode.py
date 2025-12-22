import pretty_midi
import numpy as np


def apply_sustain(pm):
    """
    Extends note durations based on Sustain Pedal (CC64) events.
    If the pedal is down, notes are held until the pedal is lifted.
    This is crucial for capturing the 'real' sound of complex piano playing.
    """
    # Create a copy to avoid modifying original object if needed elsewhere
    # (pretty_midi objects are mutable, here we mutate in place for efficiency)

    for instrument in pm.instruments:
        # Sort control changes by time
        cc_events = sorted(instrument.control_changes, key=lambda x: x.time)
        pedal_events = [cc for cc in cc_events if cc.number == 64]

        if not pedal_events:
            continue

        # Create a timeline of pedal state: (time, is_down)
        # Pedal is considered down if value >= 64
        pedal_changes = []
        is_down = False
        for cc in pedal_events:
            now_down = cc.value >= 64
            if now_down != is_down:
                pedal_changes.append((cc.time, now_down))
                is_down = now_down

        # Extend notes
        # We need to process notes that end while pedal is ON
        new_notes = []
        for note in instrument.notes:
            end_t = note.end

            # Find the pedal state at note release
            # We look for the first pedal change AFTER the note ends
            idx = -1
            for i, (t, down) in enumerate(pedal_changes):
                if t >= end_t:
                    idx = i
                    break

            # If we found a pedal event after the note end
            if idx != -1:
                # Check if pedal was DOWN before this change
                # (Look at the state of the previous event)
                prev_state_down = pedal_changes[idx - 1][1] if idx > 0 else False

                # If pedal was down at note.end, extend note to when pedal lifts
                if prev_state_down:
                    # Find the next time pedal goes UP (False)
                    lift_time = None
                    for t, down in pedal_changes[idx:]:
                        if not down:
                            lift_time = t
                            break

                    if lift_time:
                        note.end = max(note.end, lift_time)

            new_notes.append(note)

        instrument.notes = new_notes
    return pm


def velocity_to_bin(velocity, num_bins=32):
    """Quantize velocity to reduce token vocabulary size while keeping dynamics."""
    if velocity == 0: return 0
    # Map 1-127 to 1-(num_bins)
    return int(np.ceil(velocity / 127 * num_bins))


def encode_midi(midi_path, time_step_ms=10, velocity_bins=32):
    pm = pretty_midi.PrettyMIDI(midi_path)

    # 1. Preprocess: "Bake" the sustain pedal into note durations
    pm = apply_sustain(pm)

    instrument = pm.instruments[0]
    events = []

    # 2. Collect Note Events
    for note in instrument.notes:
        # Quantize velocity
        vel_bin = velocity_to_bin(note.velocity, velocity_bins)
        events.append((note.start, "ON", note.pitch, vel_bin))
        events.append((note.end, "OFF", note.pitch, None))

    # 3. Collect Tempo Events (Optional, allows capturing rubato)
    # We round tempo to nearest 4 BPM to reduce noise
    tempo_times, tempi = pm.get_tempo_changes()
    for t, bpm in zip(tempo_times, tempi):
        events.append((t, "TEMPO", int(round(bpm / 4) * 4), None))

    # 4. SORTING - The Critical Step for Complex Polyphony
    # Priority: OFF < TEMPO < ON
    # This ensures we close old notes before starting new ones at the same split-second.
    priority_map = {"OFF": 0, "TEMPO": 1, "ON": 2}

    # Sort by Time -> Priority -> Pitch
    events.sort(key=lambda x: (x[0], priority_map[x[1]], x[2] if x[2] else 0))

    tokens = []
    current_time = 0.0

    for event_time, etype, value, param in events:
        # Calculate time shift
        delta_seconds = event_time - current_time

        # We only emit shifts if the delta is significant (>= 1 step)
        if delta_seconds >= (time_step_ms / 1000):
            ms_shift = int(round(delta_seconds * 1000))
            # Quantize to grid
            steps = ms_shift // time_step_ms

            # Emit time tokens (we can stack them or use a specific vocabulary)
            # Here we stack small shifts, e.g., SHIFT_10, SHIFT_10...
            # Ideally, use a vocabulary like SHIFT_10, SHIFT_50, SHIFT_100, etc.
            # For simplicity, we stick to generic linear shifts:
            if steps > 0:
                total_shift = steps * time_step_ms
                tokens.append(f"TIME_SHIFT_{total_shift}")
                current_time += total_shift / 1000

        if etype == "ON":
            tokens.append(f"NOTE_ON_{value}_{param}")  # param is velocity bin
        elif etype == "OFF":
            tokens.append(f"NOTE_OFF_{value}")
        elif etype == "TEMPO":
            tokens.append(f"TEMPO_{value}")

    return tokens