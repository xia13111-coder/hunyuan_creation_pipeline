"""Compatibility helpers for Isaac Sim 5.x and 6.x standalone scripts."""


def get_simulation_app_class():
    """Return SimulationApp from the namespace available in this Isaac Sim."""
    try:
        from isaacsim import SimulationApp

        return SimulationApp
    except ImportError:
        pass

    try:
        from isaacsim.simulation_app import SimulationApp

        return SimulationApp
    except ImportError:
        pass

    from omni.isaac.kit import SimulationApp

    return SimulationApp
