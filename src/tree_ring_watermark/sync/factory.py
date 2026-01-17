from .base import BaseSync

def get_sync_model(sync_type: str, syncpath: str, device: str) -> BaseSync:
    if sync_type == "wam":
        from .wam import WamSync
        return WamSync(syncpath, device)
    elif sync_type == "sync_seal":
        from .sync_seal import SyncSeal
        return SyncSeal(syncpath, device)
    else:
        raise ValueError(f"Unknown sync type: {sync_type}")
