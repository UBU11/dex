import mujoco_menagerie as mm

model = mm.load("leap_hand")
spec = mm.get("leap_hand").spec()
