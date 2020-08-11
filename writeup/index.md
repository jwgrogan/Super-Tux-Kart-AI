# <b><center>TEAM 13 - FINAL PROJECT WRITEUP</center></b>

Explain your strategy here. Limit yourself to two pages, either markdown or pdf.

# Files
## player.py
## comms.py
## model.py
## train.py
## gen_data_utils.py
## get_train_data.py

# Model Design

# Model Decision Flow
## Evaluation
- Check for puck location
    - Evaluate image using model
    - Check comms if no puck location returned
- Set boolean value nearPuck

# ---ACTIONS TO GET TO PUCK---
- IF OFFENSE:
- If not nearPuck, get to puck
- If puck is left, do left turn actions
- If puck is right, do right turn actions
# IF DEFENSE:
- If not nearPuck AND puck location within some distance of goal, get to puck
- If puck is left, do left turn actions
- If puck is right, do right turn actions
# ---ACTIONS TO HIT PUCK---
- If nearPuck, angle kart towards opponent goal
- If goal is left, do left turn actions
- If goal is right, do right turn actions
- If angle of kart to goal is within some range AND puck in front of kart, hit puck


# Controller
- If multiple players on team, want to avoid two people trying to hit puck at same time
- Create a boolean value of "playerOnPuck" or some shit
- If true, bash the fuck out of AI
- If false, go after puck

# Kart Communications


# Training

![Tux, the Linux mascot](/assets/images/tux.png)