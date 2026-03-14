# AirSign

Draw your signature in the air using hand tracking. Uses your webcam to detect your index finger and renders a glowing Zima Blue trail as you write.

## Setup

```bash
pip install -r requirements.txt
python airsign.py
```

Requires Python 3.8+ and a webcam.

On macOS, grant camera access to your terminal app in **System Settings > Privacy & Security > Camera**.

## Controls

| Action | How |
|--------|-----|
| Draw | Point with index finger (other fingers curled) |
| Pause | Open hand (all fingers extended) |
| Clear canvas | Press `c` |
| Quit | Press `q` |

## Recording

Use macOS screen recording (`Cmd+Shift+5`) to capture the AirSign window while you sign.
