#!/usr/bin/env python3
import time
import yaml
import argparse
import threading
from datetime import datetime
from pynput import keyboard
import notify2

# Modifier key sets
CTRL_KEYS = {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}
ALT_KEYS  = {keyboard.Key.alt,  keyboard.Key.alt_l,  keyboard.Key.alt_r}

# Track the most recent file we saved
last_saved_file = [None]

def print_notify(message):
    print(message)
    notify("k_track", message)

def notify(title, message):
    notify2.init("Notifier")
    n = notify2.Notification(title, message)
    n.set_timeout(3000)  # 3 seconds
    n.show()

def gen_filename(prefix="keylog", ext=".yaml"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}{ext}"

def save_events(path, events):
    with open(path, 'w') as f:
        yaml.safe_dump(events, f)
    last_saved_file[0] = path
    print_notify(f"[✓] Saved {len(events)} events → {path}")

def reconstruct_events(events):
    """
    Given a list of events each with 'tmod' field (modulo 10s),
    compute a real timestamp 'ts' for each by detecting wraparounds.
    """
    rollover = 0
    prev = None
    for ev in events:
        tmod = ev['tmod']
        if prev is not None and tmod < prev:
            rollover += 1
        ev['ts'] = rollover * 10.0 + tmod
        prev = tmod
    return events

def replay_file(path):
    try:
        with open(path) as f:
            events = yaml.safe_load(f) or []
    except Exception as e:
        print_notify(f"[!] Failed to load {path}: {e}")
        return

    if not events:
        print_notify("[!] No events to replay.")
        return

    events = reconstruct_events(events)
    # ensure sorted by real time
    events.sort(key=lambda e: e['ts'])

    controller = keyboard.Controller()
    print_notify(f"[⇦] Replaying {path} …")

    pressed = set()
    prev_ts = 0.0

    for ev in events:
        dt = ev['ts'] - prev_ts
        if dt > 0:
            time.sleep(dt)

        # reconstruct key object
        if ev['key_type'] == 'char':
            k = ev['key']
        else:
            name = ev['key']
            try:
                k = getattr(keyboard.Key, name)
            except AttributeError:
                if name.startswith('ctrl'):
                    k = keyboard.Key.ctrl
                elif name.startswith('alt'):
                    k = keyboard.Key.alt
                else:
                    prev_ts = ev['ts']
                    continue

        # press or release
        if ev['type'] == 'down':
            controller.press(k)
            pressed.add(k)
        else:
            controller.release(k)
            pressed.discard(k)

        prev_ts = ev['ts']

    # release any stuck keys
    for k in list(pressed):
        controller.release(k)
    if pressed:
        print_notify(f"[!] Released stuck keys: {pressed}")
    pressed.clear()

    print_notify("[⇦] Replay complete.")

def record(initial_output):
    output_file = initial_output or gen_filename()
    print_notify(f"[→] Starting recording → {output_file}")

    events = []
    recording = True
    pressed = set()
    t0 = time.time()

    def on_press(key):
        nonlocal recording, events, t0, output_file
        pressed.add(key)

        # Pause & save: Ctrl+Alt+5
        if (isinstance(key, keyboard.KeyCode) and
            key.char == '5' and
            pressed & CTRL_KEYS and
            pressed & ALT_KEYS and
            recording):
            save_events(output_file, events)
            recording = False
            print_notify("[‖] Paused. Press Ctrl+Alt+9 to resume into a new file.")
            return

        # Resume new file: Ctrl+Alt+9
        if (isinstance(key, keyboard.KeyCode) and
            key.char == '9' and
            pressed & CTRL_KEYS and
            pressed & ALT_KEYS and
            not recording):
            output_file = gen_filename()
            events.clear()
            t0 = time.time()
            recording = True
            print_notify(f"[→] Resuming recording → {output_file}")
            return

        # Replay last saved: Ctrl+Alt+7
        if (isinstance(key, keyboard.KeyCode) and
            key.char == '7' and
            pressed & CTRL_KEYS and
            pressed & ALT_KEYS):
            if last_saved_file[0]:
                print_notify(f"[⇦] Launching replay of → {last_saved_file[0]}")
                threading.Thread(
                    target=replay_file,
                    args=(last_saved_file[0],),
                    daemon=True
                ).start()
            else:
                print_notify("[!] No saved file to replay yet.")
            return

        # Record down event (modulo time)
        if recording:
            now = time.time()
            tmod = (now - t0) % 10.0
            if isinstance(key, keyboard.KeyCode):
                events.append({
                    'tmod': tmod, 'type': 'down',
                    'key_type': 'char',
                    'key': key.char, 'vk': key.vk
                })
            else:
                events.append({
                    'tmod': tmod, 'type': 'down',
                    'key_type': 'special',
                    'key': key.name
                })

    def on_release(key):
        nonlocal events
        pressed.discard(key)

        if recording:
            now = time.time()
            tmod = (now - t0) % 10.0
            if isinstance(key, keyboard.KeyCode):
                events.append({
                    'tmod': tmod, 'type': 'up',
                    'key_type': 'char',
                    'key': key.char, 'vk': key.vk
                })
            else:
                events.append({
                    'tmod': tmod, 'type': 'up',
                    'key_type': 'special',
                    'key': key.name
                })

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print_notify(
        "🎙 Recording…\n"
        "  • Ctrl+Alt+5 → pause & save\n"
        "  • Ctrl+Alt+9          → resume into a new timestamped file\n"
        "  • Ctrl+Alt+7          → replay last saved file\n"
        "  • Ctrl+C              → exit & final save\n"
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # flush on exit
        if recording and events:
            save_events(output_file, events)
        print_notify("👋 Exiting.")
        listener.stop()

def main():
    parser = argparse.ArgumentParser(
        description="Record keypresses to YAML with modulo‐10s timestamps, pause/resume & live replay"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-r', '--replay',
        help='YAML file to replay once (then exit)'
    )
    parser.add_argument(
        'output', nargs='?',
        help='(optional) initial YAML file to save events'
    )
    args = parser.parse_args()

    if args.replay:
        replay_file(args.replay)
    else:
        record(args.output)

if __name__ == '__main__':
    main()

