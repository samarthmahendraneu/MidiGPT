import pretty_midi
import numpy as np


def bin_to_velocity(bin_val, num_bins=32):
    """Convert velocity bin back to MIDI velocity (0-127)."""
    if bin_val == 0: return 0
    return int(min(127, bin_val * (127 / num_bins)))


def decode_tokens(tokens, output_path, time_step_ms=10, velocity_bins=32):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    current_time = 0.0
    active_notes = {}  # {pitch: (start_time, velocity)}

    # Tempo tracking
    tempo_times = []
    tempo_values = []

    for token in tokens:
        parts = token.split("_")
        event_type = parts[0]

        if event_type == "TIME":
            # TIME_SHIFT_100
            shift_ms = int(parts[2])
            current_time += shift_ms / 1000

        elif event_type == "NOTE":
            action = parts[1]  # ON or OFF
            pitch = int(parts[2])

            if action == "ON":
                vel_bin = int(parts[3])
                velocity = bin_to_velocity(vel_bin, velocity_bins)

                # If note is already playing, cut it off (implicit Note Off)
                # This handles edge cases where a Note Off token might be missing
                if pitch in active_notes:
                    start, old_vel = active_notes.pop(pitch)
                    # Don't add extremely short notes (artifacts)
                    if current_time - start > 0.005:
                        instrument.notes.append(pretty_midi.Note(
                            velocity=old_vel, pitch=pitch, start=start, end=current_time
                        ))

                # Start new note
                active_notes[pitch] = (current_time, velocity)

            elif action == "OFF":
                if pitch in active_notes:
                    start, vel = active_notes.pop(pitch)
                    instrument.notes.append(pretty_midi.Note(
                        velocity=vel, pitch=pitch, start=start, end=current_time
                    ))

        elif event_type == "TEMPO":
            bpm = int(parts[1])
            tempo_times.append(current_time)
            tempo_values.append(bpm)

    # Cleanup: Close any remaining active notes at the very end
    for pitch, (start, vel) in active_notes.items():
        instrument.notes.append(pretty_midi.Note(
            velocity=vel, pitch=pitch, start=start, end=current_time
        ))

    pm.instruments.append(instrument)

    # Apply tempo changes
    # pretty_midi expects arrays for tempo changes
    if tempo_times:
        # pretty_midi write function handles the formatting
        # We manually overwrite the internal tempo change arrays
        pm._tempo_changes = (tempo_times, tempo_values)

    pm.write(output_path)
    return pm