import csv
import numpy as np

class StatsTracker:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.global_data = {
            "longest_epoch": 0,
            "champion_agent": None,
            "avg_epoch_length": [],
            "avg_death_age": [],
            "avg_eat": [],
            "avg_drink": [],
            "avg_pain_pct": [],
            "avg_pain_time": [],
        }
        self.epoch_data = []
        self.current_epoch = None

    # --- Epoch Lifecycle ---
    def start_epoch(self, epoch, agents):
        self.current_epoch = {
            "epoch": epoch + 1,
            "start_avg_happiness": np.mean([a.happiness for a in agents]),
            "end_avg_happiness": None,
            "happiness_max": 0,
            "happiness_min": 1e9,
            "epoch_length": 0,
            "pain_pct": 0,
            "avg_pain_time": 0,
            "normalized_pain_time": 0,
            "champion_agent": None,
        }

    def end_epoch(self, epoch, agents, steps):
        pain_agents = [a for a in agents if a.was_in_pain]
        pain_pct = len(pain_agents) / len(agents) if agents else 0
        avg_pain_time = np.mean([a.steps_in_pain for a in agents]) if agents else 0
        normalized_pain_time = avg_pain_time / (pain_pct or 1)

        all_happiness = [a.happiness_total / max(1, a.happiness_count) for a in agents]
        self.current_epoch.update({
            "epoch_length": steps,
            "end_avg_happiness": np.mean(all_happiness),
            "happiness_max": np.max(all_happiness),
            "happiness_min": np.min(all_happiness),
            "pain_pct": pain_pct * 100,
            "avg_pain_time": avg_pain_time,
            "normalized_pain_time": normalized_pain_time,
        })

        self.epoch_data.append(self.current_epoch)
        self._update_global(agents, steps)
        self.current_epoch = None

    # --- Global Aggregation ---
    def _update_global(self, agents, steps):
        self.global_data["avg_epoch_length"].append(steps)
        self.global_data["avg_death_age"].append(np.mean([a.death_step or steps for a in agents]))
        self.global_data["avg_eat"].append(np.mean([a.times_ate for a in agents]))
        self.global_data["avg_drink"].append(np.mean([a.times_drank for a in agents]))
        self.global_data["avg_pain_pct"].append(np.mean([a.was_in_pain for a in agents]) * 100)
        self.global_data["avg_pain_time"].append(np.mean([a.steps_in_pain for a in agents]))

        if steps > self.global_data["longest_epoch"]:
            self.global_data["longest_epoch"] = steps
            # Optional: capture champion agent reference
            self.global_data["champion_agent"] = max(agents, key=lambda a: a.happiness)

    # --- Reporting ---
    def print_status(self, complexity=1):
        epoch = len(self.epoch_data)
        last = self.epoch_data[-1]
        progress = epoch / self.total_epochs * 100
        print(f"\n--- Epoch {epoch}/{self.total_epochs} ({progress:.1f}%) ---")
        print(f"Length: {last['epoch_length']} | Pain %: {last['pain_pct']:.1f} | Avg Happiness: {last['end_avg_happiness']:.3f}")
        
        if complexity >= 2:
            avg_len = np.mean(self.global_data['avg_epoch_length'])
            print(f"Global Avg Length: {avg_len:.1f} | Longest: {self.global_data['longest_epoch']}")
        if complexity >= 3:
            print(f"Pain Avg: {np.mean(self.global_data['avg_pain_pct']):.1f}% | "
                  f"Avg Pain Time: {np.mean(self.global_data['avg_pain_time']):.2f}")
        if complexity >= 4 and self.global_data['champion_agent']:
            champ = self.global_data['champion_agent']
            print(f"Champion Happiness: {champ.happiness:.2f}, Ate: {champ.times_ate}, Drank: {champ.times_drank}")

    # --- Export ---
    def export(self, filename):
        keys = list(self.epoch_data[0].keys())
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.epoch_data)
        print(f"\n📊 Stats exported to {filename}")
