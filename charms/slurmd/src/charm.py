#!/usr/bin/env python3
# Copyright 2020-2024 Omnivector, LLC.
# See LICENSE file for licensing details.

"""Slurmd Operator Charm."""

import logging
from pathlib import Path
from typing import Any, Dict, cast

from interface_slurmctld import Slurmctld, SlurmctldAvailableEvent
from ops import (
    ActionEvent,
    ActiveStatus,
    BlockedStatus,
    CharmBase,
    ConfigChangedEvent,
    InstallEvent,
    MaintenanceStatus,
    StoredState,
    UpdateStatusEvent,
    WaitingStatus,
    EventBase,
    Handle,
    main,
)
from slurmutils import calculate_rs
from slurmutils.models.option import NodeOptionSet, PartitionOptionSet
from utils import gpu, machine, nhc, rdma, service

from charms.hpc_libs.v0.slurm_ops import SlurmdManager, SlurmOpsError
from charms.operator_libs_linux.v0.juju_systemd_notices import (  # type: ignore[import-untyped]
    ServiceStartedEvent,
    ServiceStoppedEvent,
    SystemdNotices,
)

logger = logging.getLogger(__name__)

class NodeStateChangedEvent(EventBase):
    """Emitted when the slurmd node state changed."""

    def __init__(self, handle: Handle, new_state: str, reason: str, units: set(int)):
        super().__init__(handle)

        self.new_state = new_state
        self.reason = reason
        self.units = units

    def snapshot(self):
        """Snapshot the event data."""
        return {
            "new_state": self.new_state,
            "reason": self.reason,
            "units": list(self.units)
        }

    def restore(self, snapshot):
        """Restore the snapshot of the event data."""
        self.new_state = snapshot.get("new_state")
        self.reason = snapshot.get("reason")
        self.units = set(snapshot.get("units"))


class SlurmdCharm(CharmBase):
    """Slurmd lifecycle events."""

    _stored = StoredState()

    def __init__(self, *args, **kwargs):
        """Init _stored attributes and interfaces, observe events."""
        super().__init__(*args, **kwargs)

        self.on.define_event("node_state_changed", NodeStateChangedEvent)

        self._stored.set_default(
            munge_key=str(),
            new_node=True,
            nhc_conf=str(),
            nhc_params=str(),
            slurm_installed=False,
            slurmctld_available=False,
            slurmctld_host=str(),
            user_supplied_node_parameters={},
            user_supplied_partition_parameters={},
        )

        self._slurmd = SlurmdManager(snap=False)
        self._slurmctld = Slurmctld(self, "slurmctld")
        self._systemd_notices = SystemdNotices(self, ["slurmd"])

        event_handler_bindings = {
            self.on.install: self._on_install,
            self.on.update_status: self._on_update_status,
            self.on.config_changed: self._on_config_changed,
            self._slurmctld.on.slurmctld_available: self._on_slurmctld_available,
            self._slurmctld.on.slurmctld_unavailable: self._on_slurmctld_unavailable,
            self.on.service_slurmd_started: self._on_slurmd_started,
            self.on.service_slurmd_stopped: self._on_slurmd_stopped,
            self.on.set_state_action: self._on_set_state_action,
            self.on.node_config_action: self._on_node_config_action_event,
            self.on.node_state_changed: self._on_node_state_changed,
        }
        for event, handler in event_handler_bindings.items():
            self.framework.observe(event, handler)

    def _on_install(self, event: InstallEvent) -> None:
        """Perform installation operations for slurmd."""
        # Account for case where base image has been auto-upgraded by Juju and a reboot is pending
        # before charm code runs. Reboot "now", before the current hook completes, and restart the
        # hook after reboot. Prevents issues such as drivers/kernel modules being installed for a
        # running kernel pending replacement by a newer version on reboot.
        self._reboot_if_required(now=True)

        self.unit.status = MaintenanceStatus("Installing slurmd")

        try:
            self._slurmd.install()

            self.unit.status = MaintenanceStatus("Installing nhc")
            nhc.install()

            self.unit.status = MaintenanceStatus("Installing RDMA packages")
            rdma.install()

            self.unit.status = MaintenanceStatus("Detecting if machine is GPU-equipped")
            gpu_enabled = gpu.autoinstall()
            if gpu_enabled:
                self.unit.status = MaintenanceStatus("Successfully installed GPU drivers")
            else:
                self.unit.status = MaintenanceStatus("No GPUs found. Continuing")

            self.unit.set_workload_version(self._slurmd.version())
            # TODO: https://github.com/orgs/charmed-hpc/discussions/10 -
            #  Evaluate if we should continue doing the service override here
            #  for `juju-systemd-notices`.
            service.override_service()
            self._systemd_notices.subscribe()

            self._slurmd.service.enable()

            self._stored.slurm_installed = True
        except (SlurmOpsError, gpu.GPUOpsError) as e:
            logger.error(e.message)
            event.defer()

        self._check_status()
        self._reboot_if_required()

    def _on_config_changed(self, _: ConfigChangedEvent) -> None:
        """Handle charm configuration changes."""
        # Casting the type to str is required here because `get` returns a looser
        # type than what `nhc.generate_config(...)` allows to be passed.
        if nhc_conf := cast(str, self.model.config.get("nhc-conf", "")):
            if nhc_conf != self._stored.nhc_conf:
                self._stored.nhc_conf = nhc_conf
                nhc.generate_config(nhc_conf)

        user_supplied_partition_parameters = self.model.config.get("partition-config")

        if self.model.unit.is_leader():
            if user_supplied_partition_parameters is not None:
                try:
                    tmp_params = {
                        item.split("=")[0]: item.split("=")[1]
                        for item in str(user_supplied_partition_parameters).split()
                    }
                except IndexError:
                    logger.error(
                        "Error parsing partition-config. Please use KEY1=VALUE KEY2=VALUE."
                    )
                    return

                # Validate the user supplied params are valid params.
                for parameter in tmp_params:
                    if parameter not in list(PartitionOptionSet.keys()):
                        logger.error(
                            f"Invalid user supplied partition configuration parameter: {parameter}."
                        )
                        return

                self._stored.user_supplied_partition_parameters = tmp_params

                if self._slurmctld.is_joined:
                    self._slurmctld.set_partition()

    def _on_update_status(self, _: UpdateStatusEvent) -> None:
        """Handle update status."""
        self._check_status()

    def _on_slurmctld_available(self, event: SlurmctldAvailableEvent) -> None:
        """Retrieve the slurmctld_available event data and store in charm state."""
        if self._stored.slurm_installed is not True:
            event.defer()
            return

        if (slurmctld_host := event.slurmctld_host) != self._stored.slurmctld_host:
            if slurmctld_host is not None:
                self._slurmd.config_server = f"{slurmctld_host}:6817"
                self._stored.slurmctld_host = slurmctld_host
                logger.debug(f"slurmctld_host={slurmctld_host}")
            else:
                logger.debug("'slurmctld_host' not in event data.")
                return

        if (munge_key := event.munge_key) != self._stored.munge_key:
            if munge_key is not None:
                self._stored.munge_key = munge_key
                self._slurmd.munge.key.set(munge_key)
            else:
                logger.debug("'munge_key' not in event data.")
                return

        if (nhc_params := event.nhc_params) != self._stored.nhc_params:
            if nhc_params is not None:
                self._stored.nhc_params = nhc_params
                nhc.generate_wrapper(nhc_params)
                logger.debug(f"nhc_params={nhc_params}")
            else:
                logger.debug("'nhc_params' not in event data.")
                return

        logger.debug("#### Storing slurmctld_available event relation data in charm StoredState.")
        self._stored.slurmctld_available = True

        # Restart munged and slurmd after we write the event data to their respective locations.
        try:
            self._slurmd.munge.service.restart()
            logger.debug("restarted munge successfully")
        except SlurmOpsError as e:
            logger.error("failed to restart munge")
            logger.error(e.message)

        if self._slurmd.service.active():
            self._slurmd.service.restart()
        else:
            self._slurmd.service.start()

        self._check_status()

    def _on_slurmctld_unavailable(self, _) -> None:
        """Stop slurmd and set slurmctld_available = False when we lose slurmctld."""
        logger.debug("## Slurmctld unavailable")
        self._stored.slurmctld_available = False
        self._stored.nhc_params = ""
        self._stored.munge_key = ""
        self._stored.slurmctld_host = ""
        self._slurmd.service.stop()
        self._check_status()

    def _on_slurmd_started(self, _: ServiceStartedEvent) -> None:
        """Handle event emitted by systemd after slurmd daemon successfully starts."""
        self.unit.status = ActiveStatus()

    def _on_slurmd_stopped(self, _: ServiceStoppedEvent) -> None:
        """Handle event emitted by systemd after slurmd daemon is stopped."""
        self.unit.status = BlockedStatus("slurmd not running")

    def _on_set_state_action(self, event: ActionEvent) -> None:
        """Set the node state of a set of units."""
        if not self.unit.is_leader():
            event.fail("this action can only be run from the leader unit")
            return

        new_state: str = event.params["state"]
        reason: str = event.params.get("reason", "")

        if not (nodes := event.params.get("nodes")):
            self.on.node_state_changed.emit(new_state=new_state, reason=reason, units=set())
            return

        units = set()
        for node_range in nodes.split(","):
            node_range = node_range.split("-")
            length = len(node_range)
            try:
                match len(node_range):
                    case 1:
                        unit = int(node_range[0])
                        units.add(unit)
                    case 2:
                        start, end = int(node_range[0]), int(node_range[1])
                        if start > end:
                            start, end = end, start
                        units.update(range(start, end + 1))
                    case _:
                        event.fail("invalid syntax for node range")
                        return
            except ValueError as e:
                event.fail("{e}")
                return

        self.on.node_state_changed.emit(new_state=new_state, reason=reason, units=units)

    def _on_node_state_changed(self, event: NodeStateChangedEvent) -> None:
        """Set the node state of the current unit."""
        unit_number = self.unit.name.split("/", 1)[1]
        if event.units and not unit_number in event.units:
            return

        # Trigger reconfiguration of slurmd node.
        self._node_state = event.new_state
        self._node_state_reason = event.reason
        self._slurmctld.set_node()
        self._slurmd.service.restart()
        logger.debug("### Transitioned node `%s` to state `%s` with reason `%s", self.unit.name, event.new_state, event.reason)

    def _on_show_nhc_config(self, event: ActionEvent) -> None:
        """Show current nhc.conf."""
        try:
            event.set_results({"nhc.conf": nhc.get_config()})
        except FileNotFoundError:
            event.set_results({"nhc.conf": "/etc/nhc/nhc.conf not found."})

    def _on_node_config_action_event(self, event: ActionEvent) -> None:
        """Get or set the user_supplied_node_config.

        Return the node config if the `node-config` parameter is not specified, otherwise
        parse, validate, and store the input of the `node-config` parameter in stored state.
        Lastly, update slurmctld if there are updates to the node config.
        """
        valid_config = True
        config_supplied = False

        if (user_supplied_node_parameters := event.params.get("parameters")) is not None:
            config_supplied = True

            # Parse the user supplied node-config.
            node_parameters_tmp = {}
            try:
                node_parameters_tmp = {
                    item.split("=")[0]: item.split("=")[1]
                    for item in user_supplied_node_parameters.split()
                }
            except IndexError:
                logger.error(
                    "Invalid node parameters specified. Please use KEY1=VAL KEY2=VAL format."
                )
                valid_config = False

            # Validate the user supplied params are valid params.
            for param in node_parameters_tmp:
                if param not in list(NodeOptionSet.keys()):
                    logger.error(f"Invalid user supplied node parameter: {param}.")
                    valid_config = False

            # Validate the user supplied params have valid keys.
            for k, v in node_parameters_tmp.items():
                if v == "":
                    logger.error(f"Invalid user supplied node parameter: {k}={v}.")
                    valid_config = False

            if valid_config:
                if (node_parameters := node_parameters_tmp) != self._user_supplied_node_parameters:
                    self._user_supplied_node_parameters = node_parameters
                    self._slurmctld.set_node()

        results = {
            "node-parameters": " ".join(
                [f"{k}={v}" for k, v in self.get_node()["node_parameters"].items()]
            )
        }

        if config_supplied is True:
            results["user-supplied-node-parameters-accepted"] = f"{valid_config}"

        event.set_results(results)

    @property
    def hostname(self) -> str:
        """Return the hostname."""
        return self._slurmd.hostname

    @property
    def _user_supplied_node_parameters(self) -> dict[Any, Any]:
        """Return the user_supplied_node_parameters from stored state."""
        return self._stored.user_supplied_node_parameters  # type: ignore[return-value]

    @_user_supplied_node_parameters.setter
    def _user_supplied_node_parameters(self, node_parameters: dict) -> None:
        """Set the node_parameters in stored state."""
        self._stored.user_supplied_node_parameters = node_parameters

    @property
    def _node_state(self) -> str:
        """Get the node state from stored state."""
        return self._stored.node_state or "DOWN"

    @_node_state.setter
    def _node_state(self, new_state: str) -> None:
        """Set the new_node in stored state."""
        self._stored.node_state = new_state
    
    @property
    def _node_state_reason(self) -> str:
        """Get the node state reason from stored state."""
        return self._stored.node_state_reason or ""

    @_node_state_reason.setter
    def _node_state_reason(self, reason: str) -> None:
        """Set the node state reason in stored state."""
        self._stored.node_state_reason = reason

    def _check_status(self) -> bool:
        """Check if we have all needed components.

        - slurmd installed
        - slurmctld available and working
        - munge key configured and working
        """
        if self._stored.slurm_installed is not True:
            self.unit.status = BlockedStatus("Install failed. See `juju debug-log` for details")
            return False

        if self._slurmctld.is_joined is not True:
            self.unit.status = BlockedStatus("Need relations: slurmctld")
            return False

        if self._stored.slurmctld_available is not True:
            self.unit.status = WaitingStatus("Waiting on: slurmctld")
            return False

        # TODO: https://github.com/charmed-hpc/hpc-libs/issues/18 -
        #   Re-enable munge key validation check check when supported by `slurm_ops` charm library.
        # if not self._slurmd.check_munged():
        #     self.unit.status = BlockedStatus("Error configuring munge key")
        #     return False

        return True

    def _reboot_if_required(self, now: bool = False) -> None:
        """Perform a reboot of the unit if required, e.g. following a driver installation."""
        if Path("/var/run/reboot-required").exists():
            logger.info("rebooting unit %s", self.unit.name)
            self.unit.reboot(now)

    def get_node(self, state: str) -> Dict[Any, Any]:
        """Get the node from stored state."""
        slurmd_info = machine.get_slurmd_info()
        slurmd_info["NodeName"] = self.unit.name.replace('/', '-')

        gres_info = []
        if gpus := gpu.get_all_gpu():
            for model, devices in gpus.items():
                # Build gres.conf line for this GPU model.
                if len(devices) == 1:
                    device_suffix = devices[0]
                else:
                    # Get numeric range of devices associated with this GRES resource. See:
                    # https://slurm.schedmd.com/gres.conf.html#OPT_File
                    device_suffix = calculate_rs(devices)
                gres_line = {
                    "Name": "gpu",
                    "Type": model,
                    "File": f"/dev/nvidia{device_suffix}",
                }
                gres_info.append(gres_line)
                slurmd_info["Gres"] = cast(list[str], slurmd_info.get("Gres", [])) + [
                    f"gpu:{model}:{len(devices)}"
                ]

        node = {
            "node_parameters": {
                **slurmd_info,
                "MemSpecLimit": "1024",
                **self._user_supplied_node_parameters,
            },
            "node_state": {
                "state": self._node_state,
                "reason": self._node_state_reason
            }
            # Do not include GRES configuration if no GPUs detected.
            **({"gres": gres_info} if len(gres_info) > 0 else {}),
        }
        logger.debug(f"Node Configuration: {node}")
        return node

    def get_partition(self) -> Dict[Any, Any]:
        """Return the partition."""
        partition = {self.app.name: {**{"State": "UP"}, **self._stored.user_supplied_partition_parameters}}  # type: ignore[dict-item]
        logger.debug(f"partition={partition}")
        return partition


if __name__ == "__main__":  # pragma: nocover
    main.main(SlurmdCharm)
