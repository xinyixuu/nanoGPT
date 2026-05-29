#!/usr/bin/env python3
import time
import yaml
import argparse
import threading
from datetime import datetime
from pynput import keyboard
from pynput.mouse import Listener as MouseListener, Controller as MouseController, Button
import notify2

# Modifier key sets
CTRL_KEYS = {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}
ALT_KEYS  = {keyboard.Key.alt,  keyboard.Key.alt_l,  keyboard.Key.alt_r}

# Track the most recent file we saved
last_saved_file = [None]

# Screen Messages
def print_notify(message):
    print(message)
    notify("k_track", message)

def notify(title, message):
    notify2.init("Notifier")
    n = notify2.Notification(title, message)
    n.set_timeout(3000)  # 3 seconds
    n.show()

# --- Filename generation & saving ---
def gen_filename(prefix="keylog", ext=".yaml"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}{ext}"


def save_events(path, events):
    with open(path, 'w') as f:
        yaml.safe_dump(events, f)
    last_saved_file[0] = path
    print_notify(f"[✓] Saved {len(events)} events → {path}")

# --- Reconstruct real timestamps from modulo times ---
def reconstruct_events(events):
    rollover = 0
    prev = None
    for ev in events:
        tmod = ev['tmod']
        if prev is not None and tmod < prev:
            rollover += 1
        ev['ts'] = rollover * 10.0 + tmod
        prev = tmod
    return events

# --- Core replay logic for both keyboard and mouse ---
def replay_events(events):
    # Reconstruct and sort by real time
    events = reconstruct_events(events)
    events.sort(key=lambda e: e['ts'])

    kcontroller = keyboard.Controller()
    mcontroller = MouseController()
    print_notify("[⇦] Replaying events …")

    pressed_keys = set()
    prev_ts = 0.0

    for ev in events:
        dt = ev['ts'] - prev_ts
        if dt > 0:
            time.sleep(dt)

        device = ev.get('device')
        # Keyboard events
        if device == 'keyboard':
            # reconstruct key
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
                kcontroller.press(k)
                pressed_keys.add(k)
            else:
                kcontroller.release(k)
                pressed_keys.discard(k)

        # Mouse events
        elif device == 'mouse':
            if ev['type'] == 'move':
                mcontroller.position = (ev['x'], ev['y'])
            elif ev['type'] in ('down', 'up'):
                try:
                    btn = getattr(Button, ev['button'])
                except AttributeError:
                    btn = getattr(Button, ev['button'].lower(), None)
                if btn:
                    if ev['type'] == 'down':
                        mcontroller.press(btn)
                    else:
                        mcontroller.release(btn)
            elif ev['type'] == 'scroll':
                mcontroller.scroll(ev['dx'], ev['dy'])

        prev_ts = ev['ts']

    # release any stuck keys
    for k in list(pressed_keys):
        kcontroller.release(k)
    if pressed_keys:
        print(f"[!] Released stuck keys: {pressed_keys}")
    print_notify("[⇦] Replay complete.")

# --- File-based replay wrapper ---
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

    print_notify(f"[⇦] Replaying {path} …")
    replay_events(events)

# --- Recording logic for keyboard & mouse ---
def record(initial_output):
    output_file = initial_output or gen_filename()
    print_notify(f"[→] Starting recording → {output_file}")

    events = []
    recording = True
    pressed = set()
    t0 = time.time()

    # Keyboard handlers
    def on_press(key):
        nonlocal recording, events, t0, output_file
        pressed.add(key)
        # pause & save
        if (isinstance(key, keyboard.KeyCode) and key.char == '5' and pressed & CTRL_KEYS and pressed & ALT_KEYS and recording):
            save_events(output_file, events)
            recording = False
            print_notify("[‖] Paused. Press Ctrl+Alt+9 to resume into a new file.")
            return
        # resume new file
        if (isinstance(key, keyboard.KeyCode) and key.char == '9' and pressed & CTRL_KEYS and pressed & ALT_KEYS and not recording):
            output_file = gen_filename()
            events.clear(); t0 = time.time(); recording = True
            print_notify(f"[→] Resuming recording → {output_file}")
            return
        # replay last saved
        if (isinstance(key, keyboard.KeyCode) and key.char == '7' and pressed & CTRL_KEYS and pressed & ALT_KEYS):
            if last_saved_file[0]: threading.Thread(target=replay_file, args=(last_saved_file[0],), daemon=True).start()
            else: print_notify("[!] No saved file to replay yet.")
            return
        # log key down
        if recording:
            tmod = (time.time() - t0) % 10.0
            if isinstance(key, keyboard.KeyCode):
                events.append({'tmod': tmod, 'type': 'down', 'device':'keyboard', 'key_type':'char','key':key.char,'vk':key.vk})
            else:
                events.append({'tmod': tmod, 'type': 'down', 'device':'keyboard', 'key_type':'special','key':key.name})

    def on_release(key):
        nonlocal events
        pressed.discard(key)
        if recording:
            tmod = (time.time() - t0) % 10.0
            if isinstance(key, keyboard.KeyCode):
                events.append({'tmod': tmod, 'type': 'up', 'device':'keyboard','key_type':'char','key':key.char,'vk':key.vk})
            else:
                events.append({'tmod': tmod, 'type': 'up', 'device':'keyboard','key_type':'special','key':key.name})

    # Mouse handlers
    def on_move(x, y):
        if recording:
            tmod = (time.time() - t0) % 10.0
            events.append({'tmod': tmod, 'type': 'move', 'device':'mouse','x':x,'y':y})
    def on_click(x, y, button, pressed_btn):
        if recording:
            tmod = (time.time() - t0) % 10.0
            ev_type = 'down' if pressed_btn else 'up'
            events.append({'tmod': tmod, 'type': ev_type, 'device':'mouse','button':button.name})
    def on_scroll(x, y, dx, dy):
        if recording:
            tmod = (time.time() - t0) % 10.0
            events.append({'tmod': tmod, 'type': 'scroll', 'device':'mouse','dx':dx,'dy':dy})

    # start listeners
    ml = MouseListener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
    kl = keyboard.Listener(on_press=on_press, on_release=on_release)
    ml.start(); kl.start()

    print_notify(
        "🎙 Recording…\n"
        "  • Ctrl+Alt+5 → pause & save\n"
        "  • Ctrl+Alt+9          → resume into a new timestamped file\n"
        "  • Ctrl+Alt+7          → replay last saved file\n"
        "  • Ctrl+C              → exit & final save\n"
    )

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        if recording and events: save_events(output_file, events)
        print_notify("👋 Exiting.")
        kl.stop(); ml.stop()

# --- CLI entrypoint ---
def main():
    parser = argparse.ArgumentParser(
        description="Record key & mouse to YAML with modulo‐10s timestamps, pause/resume & live replay"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-r','--replay', help='YAML file to replay once (then exit)')
    parser.add_argument('output', nargs='?', help='(optional) initial YAML file to save events')
    args = parser.parse_args()
    if args.replay: replay_file(args.replay)
    else: record(args.output)

if __name__ == '__main__':
    main()

