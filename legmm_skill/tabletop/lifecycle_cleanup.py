"""Release only this process's completed-cohort allocations and subscriptions."""
import ctypes,gc,signal
from pathlib import Path


class SafeStop(RuntimeError):pass


_stop_signal=None


def stop_requested():return _stop_signal is not None


def check_stop(boundary):
    if stop_requested():raise SafeStop(f'Stop signal {_stop_signal} handled at safe boundary: {boundary}')


def install_stop_handler():
    global _stop_signal
    _stop_signal=None
    def stop(number,frame):
        global _stop_signal
        _stop_signal=number
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)


def memory_snapshot():
    rss=int(next(row for row in Path('/proc/self/status').read_text().splitlines() if row.startswith('VmRSS:')).split()[1])*1024
    available=int(next(row for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith('MemAvailable:')).split()[1])*1024
    return dict(rss_bytes=rss,available_ram_bytes=available)


def trim_unused():
    before=memory_snapshot();gc.collect()
    libc=ctypes.CDLL(None);trim=getattr(libc,'malloc_trim',None)
    returned=None if trim is None else int(trim(0))
    return dict(before=before,after=memory_snapshot(),malloc_trim_returned=returned)


def clear_planner_cache(planner):
    # During rejection sampling these caches contain mostly rejected candidates.
    # The live plan's configure() always recreates the complete true model;
    # no model dimensions, pose, IK, limits or collision computation change.
    counts={}
    for name in ('full_cache','model_cache','models','data'):
        cache=getattr(planner,name,None)
        if cache is not None:counts[name]=len(cache);cache.clear()
    for name in ('m','d'):
        if hasattr(planner,name):setattr(planner,name,None)
    return counts


def detach_owned_sensors(scene):
    counts={}
    for name,sensor in scene.sensors.items():
        annotators=getattr(sensor,'_annotators',{})
        counts[name]=len(annotators)
        for annotator in annotators.values():annotator.detach(sensor.render_product_paths)
        if annotators:sensor._annotators={}
        sensor._clear_callbacks()
    return counts


def release_ready_observer(simulation):
    # Installed core clear_instance clears legacy callbacks but omits this new
    # bound-method observer. Dropping this owned guard unsubscribes it, so it
    # cannot retain the completed SimulationContext across new stages.
    present=getattr(simulation,'_on_post_physics_ready_callback',None) is not None
    if present:simulation._on_post_physics_ready_callback=None
    return present
