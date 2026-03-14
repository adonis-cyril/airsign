# AirSign

Draw your signature in the air using hand tracking. Uses your webcam to detect your index finger and renders a glowing Zima Blue trail as you write.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python airsign.py
```

On first run, the hand landmarker model (~8MB) is downloaded automatically.

Requires Python 3.8+ and a webcam.

**macOS:** Grant camera access to Terminal (or your IDE) in **System Settings → Privacy & Security → Camera**, or the app will report "Could not open camera."

## Controls

| Action | How |
|--------|-----|
| Draw | Point with index finger (other fingers curled) |
| Pause | Open hand (all fingers extended) |
| Clear canvas | Press `c` |
| Quit | Press `q` |

## Recording

Use macOS screen recording (`Cmd+Shift+5`) to capture the AirSign window while you sign.
