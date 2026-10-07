HIERARCHICAL TRAFFIC SIGNAL RL

Architecture
------------
SUMO traffic
    -> Intersection Agents A/B/C
    -> clean local states
    -> Central Coordination Actor-Critic
    -> actions for A/B/C
    -> SUMO traffic lights
    -> new traffic state
    -> reward/learning

1. Put this folder on your Desktop.
2. Open Command Prompt inside the folder.
3. Run:
   build_network.bat
4. Test the network:
   run_test.bat
5. Install Python packages:
   pip install -r requirements.txt
6. Run the hierarchical controller:
   run_hierarchical.bat

Actions
-------
0 = keep current signal behavior
1 = extend the current green phase
2 = request a switch after the current green phase

Important
---------
The hierarchical Actor-Critic design is an implementation choice based on the
project's MARL + Actor-Critic + hierarchical direction. The uploaded project
materials do not specify an exact hierarchical algorithm or exact reward
formula.
