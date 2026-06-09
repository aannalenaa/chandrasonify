import json
from datetime import datetime
from pathlib import Path
from typing import Any


# region RunConfig
# — Run Configuration Manager ———————————————————————————————————
class RunConfig:
    """Pipeline decision logger for reproducible runs.

    Persists LLM decisions to JSON for replay mode. Load with RunConfig.load()
    to reuse decisions; create new with RunConfig(obs_id) to start fresh.
    """

    def __init__(self, obs_id: str, out_root: str = "."):
        """Initialize a new run configuration.

        Args:
            obs_id (str): Observation ID.
            out_root (str): Output root directory (default ".").
        """
        self.obs_id = obs_id
        self.replay = False
        self._data: dict[str, Any] = {
            "obs_id": obs_id,
            "created_at": datetime.now().isoformat(),
            "decisions": {},
        }
        self._path = Path(out_root) / f"output_{obs_id}" / "run_config.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _config_path(cls, obs_id: str, out_root: str = ".") -> Path:
        """Get path to run_config.json.

        Args:
            obs_id (str): Observation ID.
            out_root (str): Output root directory (default ".").

        Returns:
            Path: Full path to run_config.json.
        """
        return Path(out_root) / f"output_{obs_id}" / "run_config.json"

    @classmethod
    def _from_data(
        cls,
        obs_id: str,
        path: Path,
        data: dict[str, Any],
        replay: bool,
    ) -> "RunConfig":
        """Construct RunConfig from loaded data.

        Args:
            obs_id (str): Observation ID.
            path (Path): Path to run_config.json.
            data (dict[str, Any]): Deserialized config data.
            replay (bool): Replay mode flag.

        Returns:
            RunConfig: Initialized instance.
        """
        instance = cls.__new__(cls)
        instance.obs_id = obs_id
        instance.replay = replay
        instance._path = path
        instance._data = data
        return instance

    @classmethod
    def load(cls, obs_id: str, out_root: str = ".") -> "RunConfig":
        """Load existing config for replay.

        Args:
            obs_id (str): Observation ID.
            out_root (str): Output root directory (default ".").

        Returns:
            RunConfig: Instance in replay mode with loaded data.

        Raises:
            FileNotFoundError: If run_config.json does not exist.
        """
        path = cls._config_path(obs_id, out_root)
        if not path.exists():
            raise FileNotFoundError(f"No run config found at {path}")
        with open(path) as f:
            data = json.load(f)
        instance = cls._from_data(obs_id, path, data, replay=True)
        print(f"[RunConfig] Replaying from {instance._path}")
        print(
            "[RunConfig] Original run: "
            f"{instance._data.get('created_at', 'unknown')}"
        )
        return instance

    @classmethod
    def load_or_create(
        cls, obs_id: str, out_root: str = ".", fresh: bool = False
    ) -> "RunConfig":
        """Load existing config or create new one.

        Args:
            obs_id (str): Observation ID.
            out_root (str): Output root directory (default ".").
            fresh (bool): Start fresh without prompting (default False).

        Returns:
            RunConfig: New or loaded instance.
        """
        path = cls._config_path(obs_id, out_root)
        if path.exists():
            with open(path) as f:
                existing = json.load(f)
            instance = cls._from_data(obs_id, path, existing, replay=False)
            print(f"[RunConfig] Found existing config at {instance._path}")

            if fresh:
                # Fresh run — overwrite without prompting
                print("[RunConfig] Starting fresh run (--fresh flag set)")
                instance._data = {
                    "obs_id": obs_id,
                    "created_at": datetime.now().isoformat(),
                    "decisions": {},
                }
                instance._save()
            else:
                # Ask user
                use_existing = (
                    input("Replay existing run config? [n means fresh run] (y/n): ")
                    .strip()
                    .lower()
                )
                if use_existing == "y":
                    instance.replay = True
                    print("[RunConfig] Replay mode ON")
                else:
                    # Fresh run — overwrite
                    instance._data = {
                        "obs_id": obs_id,
                        "created_at": datetime.now().isoformat(),
                        "decisions": {},
                    }
                    instance._save()
                print("[RunConfig] Starting fresh run")
            return instance

        return cls(obs_id, out_root)

    def log(self, key: str, value: Any, source: str = "llm") -> None:
        """Log a pipeline decision.

        Args:
            key (str): Decision identifier.
            value (Any): Decision value.
            source (str): Source ('llm' | 'manual' | 'default', default 'llm').
        """
        self._data["decisions"][key] = {
            "value": value,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        }
        self._save()
        print(f"[RunConfig] Logged '{key}' (source: {source})")

    def get(self, key: str) -> Any:
        """Retrieve a logged decision.

        Args:
            key (str): Decision key.

        Returns:
            Any: The logged decision value.

        Raises:
            KeyError: If key not found.
        """
        if key not in self._data["decisions"]:
            raise KeyError(
                f"[RunConfig] Key '{key}' not found in config. "
                "Available keys: "
                f"{list(self._data['decisions'].keys())}"
            )
        entry = self._data["decisions"][key]
        print(
            f"[RunConfig] Using saved '{key}' " f"(originally from: {entry['source']})"
        )
        return entry["value"]

    def has(self, key: str) -> bool:
        return key in self._data["decisions"]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in decisions."""
        return key in self._data["decisions"]

    def _save(self) -> None:
        """Save configuration to run_config.json."""
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    def summary(self) -> None:
        """Print summary of all logged decisions."""
        print("\n[RunConfig] Decision log:")
        print(f"  Obs ID:   {self._data['obs_id']}")
        print(f"  Created:  {self._data.get('created_at', 'unknown')}")
        print(f"  Replay:   {self.replay}")
        print(f"  Config:   {self._path}")
        print()
        for key, entry in self._data["decisions"].items():
            print(f"  {key:<30} [{entry['source']}]  {entry['value']}")


# endregion
