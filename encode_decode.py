import pretty_midi
import numpy as np

def apply_sustain(pm):
    for instrument in pm.instruments:
        cc_events = sorted(
            [cc for cc in instrument.control_changes if cc.number == 64],
            key=lambda x: x.time
        )
        if not cc_events:
            continue

        pedal_changes = []
        is_down = False
        for cc in cc_events:
            now_down = cc.value >= 64
            if now_down != is_down:
                pedal_changes.append((cc.time, now_down))
                is_down = now_down

        for note in instrument.notes:
            for i, (t, down) in enumerate(pedal_changes):
                if t >= note.end:
                    prev_down = pedal_changes[i - 1][1] if i > 0 else False
                    if prev_down:
                        for t2, down2 in pedal_changes[i:]:
                            if not down2:
                                note.end = max(note.end, t2)
                                break
                    break
    return pm


# -------------------------
# ENCODE (less lossy)
# -------------------------
def encode_midi(midi_path):
    pm = pretty_midi.PrettyMIDI(midi_path)
    pm = apply_sustain(pm)

    instruments = [inst for inst in pm.instruments if not inst.is_drum]

    events = []
    inst_ids = {id(inst): i for i, inst in enumerate(instruments)}

    # program events (emit at t=0)
    for inst in instruments:
        inst_id = inst_ids[id(inst)]
        events.append((0.0, "PROG", inst_id, inst.program))

    # note events
    for inst in instruments:
        if not inst.is_drum:
            inst_id = inst_ids[id(inst)]
            for note in inst.notes:
                events.append((note.start, "INST", inst_id))
                events.append((note.start, "ON", inst_id, note.pitch, int(note.velocity)))
                events.append((note.end, "INST", inst_id))
                events.append((note.end, "OFF", inst_id, note.pitch))

    # tempo events (keep, but note: pretty_midi may not roundtrip perfectly)
    tempo_times, tempi = pm.get_tempo_changes()
    for t, bpm in zip(tempo_times, tempi):
        events.append((t, "TEMPO", int(round(bpm))))

    priority = {"TEMPO": 0, "PROG": 1, "INST": 2, "OFF": 3, "ON": 4}
    events.sort(key=lambda e: (e[0], priority[e[1]]))

    tokens = []
    current_time = 0.0
    current_inst = None

    for event in events:
        t = event[0]
        delta_ms = int(round((t - current_time) * 1000))
        if delta_ms > 0:
            tokens.append(f"TIME_SHIFT_{delta_ms}")
            current_time += delta_ms / 1000.0

        etype = event[1]
        if etype == "TEMPO":
            tokens.append(f"TEMPO_{event[2]}")
        elif etype == "PROG":
            _, _, inst_id, program = event
            tokens.append(f"PROG_{inst_id}_{program}")
        elif etype == "INST":
            inst_id = event[2]
            if inst_id != current_inst:
                tokens.append(f"INST_{inst_id}")
                current_inst = inst_id
        elif etype == "ON":
            _, _, inst_id, pitch, vel = event
            if inst_id != current_inst:
                tokens.append(f"INST_{inst_id}")
                current_inst = inst_id
            tokens.append(f"NOTE_ON_{pitch}_{vel}")
        elif etype == "OFF":
            _, _, inst_id, pitch = event
            if inst_id != current_inst:
                tokens.append(f"INST_{inst_id}")
                current_inst = inst_id
            tokens.append(f"NOTE_OFF_{pitch}")

    return tokens


# -------------------------
# DECODE (less lossy)
# -------------------------
def decode_tokens(tokens, output_path):
    pm = pretty_midi.PrettyMIDI()
    instruments = {}          # inst_id -> pretty_midi.Instrument
    programs = {}             # inst_id -> program
    active_notes = {}         # (inst_id, pitch) -> (start, velocity)
    current_inst = None
    current_time = 0.0

    tempo_times, tempo_values = [], []

    for token in tokens:
        if token.startswith("TIME_SHIFT_"):
            try:
                ms = int(token.split("_")[2])
                current_time += ms / 1000.0
            except:
                continue

        elif token.startswith("TEMPO_"):
            try:
                bpm = int(token.split("_")[1])
                tempo_times.append(current_time)
                tempo_values.append(bpm)
            except:
                continue

        elif token.startswith("PROG_"):
            # PROG_{inst_id}_{program}
            try:
                _, inst_id, program = token.split("_")
                inst_id = int(inst_id); program = int(program)
                programs[inst_id] = program
                if inst_id not in instruments:
                    instruments[inst_id] = pretty_midi.Instrument(program=program)
                else:
                    instruments[inst_id].program = program
            except:
                continue

        elif token.startswith("INST_"):
            try:
                inst_id = int(token.split("_")[1])
                current_inst = inst_id
                if inst_id not in instruments:
                    program = programs.get(inst_id, 0)
                    instruments[inst_id] = pretty_midi.Instrument(program=program)
            except:
                continue

        elif token.startswith("NOTE_") and current_inst is not None:
            parts = token.split("_")
            if len(parts) < 3:
                continue

            action = parts[1]
            pitch = int(parts[2])
            key = (current_inst, pitch)

            if action == "ON" and len(parts) == 4:
                vel = int(parts[3])

                # close re-on
                if key in active_notes:
                    s, v = active_notes.pop(key)
                    instruments[current_inst].notes.append(pretty_midi.Note(v, pitch, s, current_time))

                active_notes[key] = (current_time, vel)

            elif action == "OFF":
                if key in active_notes:
                    s, v = active_notes.pop(key)
                    instruments[current_inst].notes.append(pretty_midi.Note(v, pitch, s, current_time))

    for (inst_id, pitch), (s, v) in active_notes.items():
        instruments[inst_id].notes.append(pretty_midi.Note(v, pitch, s, current_time))

    for inst in instruments.values():
        pm.instruments.append(inst)

    # tempo map: may still not perfectly roundtrip with pretty_midi
    if tempo_times:
        pm._tempo_changes = (np.array(tempo_times), np.array(tempo_values))

    pm.write(output_path)
    return pm


tokens = encode_midi('data/sample1.mid')
print("encoded tokens:", tokens)
# ============================================================
# 1. RAW TEXT
# ============================================================


text = ' '.join(tokens)
# increase data size (safe) with spce
decode_tokens(tokens, 'data/og_output.mid')
