"""
SysPilot - AI Agent for System Monitoring and Application Tuning

An intelligent agent that monitors system resources, profiles applications,
and dynamically tunes configurations for optimal performance.
"""

__version__ = "0.1.0"
__author__ = "SysPilot Team"

from syspilot.core.agent import SysPilotAgent
from syspilot.core.config import SysPilotConfig

__all__ = ["SysPilotAgent", "SysPilotConfig", "__version__"]
