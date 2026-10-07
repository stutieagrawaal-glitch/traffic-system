import traci

SUMO_BINARY = "sumo-gui"
SUMO_CONFIG = "../sumo/simulation.sumocfg"

traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])

print("\nConnected to SUMO\n")
print("Traffic lights:", traci.trafficlight.getIDList())

for tls_id in traci.trafficlight.getIDList():
    print(f"\n{tls_id} phases:")
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    for i, phase in enumerate(logic.phases):
        print(f"  Phase {i}: state={phase.state}, duration={phase.duration}")

for step in range(300):
    traci.simulationStep()
    if step % 25 == 0:
        print("Time:", step, "Vehicles:", len(traci.vehicle.getIDList()))

traci.close()
print("\nThree-intersection test finished.")
