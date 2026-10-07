import os
import random
from collections import deque

import numpy as np
import traci
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

SUMO_BINARY = "sumo-gui"
SUMO_CONFIG = "../sumo/simulation.sumocfg"

TLS_IDS = ["A", "B", "C"]
DECISION_INTERVAL = 10
QUEUE_SPEED = 0.5
STATE_SIZE_PER_INTERSECTION = 11
GLOBAL_STATE_SIZE = len(TLS_IDS) * STATE_SIZE_PER_INTERSECTION
NUM_ACTIONS = 3

# Hierarchical design:
# 1. IntersectionAgent = local data-processing/state-building layer.
# 2. CoordinationAgent = high-level centralized Actor-Critic decision maker.
# 3. SignalController = safely applies keep/extend/switch decisions.

INCOMING = {
    "A": {
        "North": "A_N_to_A",
        "South": "A_S_to_A",
        "East": "B_A",
        "West": "W_A",
    },
    "B": {
        "North": "B_N_to_B",
        "South": "B_S_to_B",
        "East": "C_B",
        "West": "A_B",
    },
    "C": {
        "North": "C_N_to_C",
        "South": "C_S_to_C",
        "East": "E_C",
        "West": "B_C",
    },
}


def edge_metrics(edge_id):
    """Convert raw SUMO vehicle data into local traffic measurements."""
    vehicle_ids = traci.edge.getLastStepVehicleIDs(edge_id)
    queue = 0
    waiting = 0.0
    speed_sum = 0.0

    for vehicle_id in vehicle_ids:
        speed = traci.vehicle.getSpeed(vehicle_id)
        speed_sum += speed
        if speed < QUEUE_SPEED:
            queue += 1
        waiting += traci.vehicle.getWaitingTime(vehicle_id)

    count = len(vehicle_ids)
    mean_speed = speed_sum / count if count else 0.0
    return count, queue, waiting, mean_speed


class IntersectionAgent:
    """Lower layer: one intersection cleans raw traffic data into a local state."""

    def __init__(self, intersection_id):
        self.id = intersection_id

    def observe(self):
        counts = []
        queues = []
        waits = []
        speeds = []

        for direction in ["North", "South", "East", "West"]:
            count, queue, waiting, mean_speed = edge_metrics(
                INCOMING[self.id][direction]
            )
            counts.append(count)
            queues.append(queue)
            waits.append(waiting)
            speeds.append(mean_speed)

        phase = traci.trafficlight.getPhase(self.id)
        phase_count = len(traci.trafficlight.getAllProgramLogics(self.id)[0].phases)
        phase_norm = phase / max(phase_count - 1, 1)

        # Normalized clean local state.
        # Counts/queues are capped to avoid one large queue dominating training.
        state = np.array(
            [
                *[min(x, 20) / 20.0 for x in counts],
                *[min(x, 20) / 20.0 for x in queues],
                min(sum(waits), 100.0) / 100.0,
                min(np.mean(speeds), 13.89) / 13.89,
                phase_norm,
            ],
            dtype=np.float32,
        )

        raw = {
            "counts": counts,
            "queues": queues,
            "waiting": waits,
            "mean_speed": speeds,
            "phase": phase,
        }
        return state, raw


class ActorCritic(nn.Module):
    """Central coordinator: one global critic + one action head per intersection."""

    def __init__(self, input_size, num_agents, num_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.actor_heads = nn.ModuleList(
            [nn.Linear(128, num_actions) for _ in range(num_agents)]
        )
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        features = self.shared(state)
        logits = [head(features) for head in self.actor_heads]
        value = self.critic(features)
        return logits, value


class CoordinationAgent:
    """High layer: coordinates all intersections using centralized Actor-Critic."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.model = ActorCritic(
            GLOBAL_STATE_SIZE, len(TLS_IDS), NUM_ACTIONS
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.gamma = 0.95
        self.previous_state = None
        self.previous_actions = None
        self.previous_log_probs = None
        self.previous_value = None

    def choose_actions(self, global_state):
        state_tensor = torch.tensor(
            global_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        logits, value = self.model(state_tensor)
        actions = []
        log_probs = []

        for agent_logits in logits:
            distribution = Categorical(logits=agent_logits)
            action = distribution.sample()
            actions.append(int(action.item()))
            log_probs.append(distribution.log_prob(action))

        total_log_prob = torch.stack(log_probs).sum()
        return actions, total_log_prob, value.squeeze()

    def learn(self, reward, next_state, done=False):
        if self.previous_state is None:
            return

        state_tensor = torch.tensor(
            self.previous_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        next_tensor = torch.tensor(
            next_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        _, value = self.model(state_tensor)
        with torch.no_grad():
            _, next_value = self.model(next_tensor)
            target = torch.tensor(reward, dtype=torch.float32)
            if not done:
                target = target + self.gamma * next_value.squeeze()

        advantage = target - value.squeeze()
        actor_loss = -self.previous_log_probs * advantage.detach()
        critic_loss = advantage.pow(2)
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

    def remember(self, state, actions, log_probs, value):
        self.previous_state = np.asarray(state, dtype=np.float32)
        self.previous_actions = actions
        self.previous_log_probs = log_probs
        self.previous_value = value


class SignalController:
    """Applies coordinator actions without directly jumping through unsafe phases."""

    KEEP = 0
    EXTEND = 1
    SWITCH = 2

    def apply(self, tls_id, action):
        phase = traci.trafficlight.getPhase(tls_id)
        state = traci.trafficlight.getRedYellowGreenState(tls_id)

        if action == self.KEEP:
            return "KEEP"

        if action == self.EXTEND:
            # Add green time when a green phase is active.
            if "G" in state or "g" in state:
                traci.trafficlight.setPhaseDuration(tls_id, 8.0)
                return "EXTEND_GREEN"
            return "WAIT_FOR_GREEN"

        if action == self.SWITCH:
            # Do not cut yellow/all-red phases. Ask SUMO to finish the current
            # green phase soon; its programmed yellow transition then occurs.
            if "G" in state or "g" in state:
                traci.trafficlight.setPhaseDuration(tls_id, 1.0)
                return "REQUEST_SWITCH"
            return "WAIT_FOR_GREEN"

        return "UNKNOWN"


def network_reward(local_raw):
    """Network-level reward: lower total queue and waiting is better."""
    total_queue = 0
    total_wait = 0.0

    for raw in local_raw:
        total_queue += sum(raw["queues"])
        total_wait += sum(raw["waiting"])

    # Implementation choice; exact reward was not fixed in the project source.
    return -(total_queue + 0.1 * total_wait)


def main():
    print("Starting hierarchical MARL traffic controller...")
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])

    local_agents = [IntersectionAgent(tls_id) for tls_id in TLS_IDS]
    coordinator = CoordinationAgent()
    signal_controller = SignalController()

    try:
        for step in range(2000):
            traci.simulationStep()

            if step % DECISION_INTERVAL != 0:
                continue

            local_states = []
            local_raw = []

            for agent in local_agents:
                state, raw = agent.observe()
                local_states.append(state)
                local_raw.append(raw)

            global_state = np.concatenate(local_states).astype(np.float32)
            reward = network_reward(local_raw)

            actions, log_probs, value = coordinator.choose_actions(global_state)

            if coordinator.previous_state is not None:
                coordinator.learn(reward, global_state)

            coordinator.remember(
                global_state, actions, log_probs, value
            )

            results = []
            for tls_id, action in zip(TLS_IDS, actions):
                result = signal_controller.apply(tls_id, action)
                results.append(f"{tls_id}:{result}")

            queues = [sum(raw["queues"]) for raw in local_raw]
            print(
                f"Step {step:4d} | Reward {reward:8.2f} | "
                f"Queues A/B/C {queues} | "
                f"Actions {actions} | {' '.join(results)}"
            )

    finally:
        traci.close()
        print("\nHierarchical MARL simulation finished.")


if __name__ == "__main__":
    main()
