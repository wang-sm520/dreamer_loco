"""NVML process ownership without subtracting asynchronous memory snapshots."""
import math,os


def memory_bounds(total_before,total_after,processes,owner_pgid,baseline_overhead=0,get_group=os.getpgid):
    owned=[];foreign=[]
    for row in processes:
        try:ours=owner_pgid is not None and (row['pid']==owner_pgid or get_group(row['pid'])==owner_pgid)
        except ProcessLookupError:ours=row['pid']==owner_pgid
        (owned if ours else foreign).append(row)
    own=sum(row['memory_mib'] for row in owned);outside=sum(row['memory_mib'] for row in foreign)
    unknown=max(0,max(total_before,total_after)-own-outside)
    # Assign unknown transient/driver memory conservatively to our upper bound,
    # not to foreign jobs. Foreign allocations come from actual C/G PID rows.
    reserved=min(unknown,baseline_overhead)
    return dict(known_own_mib=own,known_external_mib=outside,unattributed_mib=unknown,
        owned_upper_bound_mib=own+unknown-reserved,external_upper_bound_mib=outside+reserved,
        owned_processes=owned,external_processes=foreign)


def gpu_snapshot(gpu,owner_pgid=None,baseline_overhead=0):
    import pynvml as nv
    nv.nvmlInit()
    try:
        handle=nv.nvmlDeviceGetHandleByIndex(gpu);uuid=nv.nvmlDeviceGetUUID(handle)
        if isinstance(uuid,bytes):uuid=uuid.decode()
        before=nv.nvmlDeviceGetMemoryInfo(handle)
        def allocations(device,getter):
            values={}
            for process in getter(device):
                used=int(process.usedGpuMemory)
                if used>=1<<60:raise RuntimeError('NVML cannot audit process GPU memory')
                mib=math.ceil(used/1024**2);pid=int(process.pid)
                values[pid]=max(values.get(pid,0),mib)
            return values
        def rows(device,device_uuid):
            merged=allocations(device,nv.nvmlDeviceGetComputeRunningProcesses)
            for pid,mib in allocations(device,nv.nvmlDeviceGetGraphicsRunningProcesses).items():
                merged[pid]=max(merged.get(pid,0),mib)
            return [dict(uuid=device_uuid,pid=pid,memory_mib=mib) for pid,mib in merged.items()]
        processes=rows(handle,uuid);after=nv.nvmlDeviceGetMemoryInfo(handle)
        first=math.ceil(before.used/1024**2);last=math.ceil(after.used/1024**2)
        bounds=memory_bounds(first,last,processes,owner_pgid,baseline_overhead)
        wrong=[];passive=[]
        if owner_pgid is not None:
            for index in range(nv.nvmlDeviceGetCount()):
                if index==gpu:continue
                other=nv.nvmlDeviceGetHandleByIndex(index);other_uuid=nv.nvmlDeviceGetUUID(other)
                if isinstance(other_uuid,bytes):other_uuid=other_uuid.decode()
                compute=allocations(other,nv.nvmlDeviceGetComputeRunningProcesses)
                graphics=allocations(other,nv.nvmlDeviceGetGraphicsRunningProcesses)
                for pid in sorted(compute.keys() | graphics.keys()):
                    try:ours=pid==owner_pgid or os.getpgid(pid)==owner_pgid
                    except ProcessLookupError:ours=pid==owner_pgid
                    if not ours:continue
                    compute_mib=compute.get(pid,0);graphics_mib=graphics.get(pid,0)
                    row=dict(index=index,uuid=other_uuid,pid=pid,memory_mib=max(compute_mib,graphics_mib),
                             compute_mib=compute_mib,graphics_mib=graphics_mib)
                    if compute_mib>0 or graphics_mib>32:
                        wrong.append(row)
                    elif graphics_mib>0:
                        passive.append(row)
        return dict(index=gpu,uuid=uuid,used_mib=max(first,last),free_mib=min(before.free,after.free)//1024**2,
            total_before_mib=first,total_after_mib=last,processes=processes,
            baseline_overhead_mib=max(0,max(first,last)-sum(row['memory_mib'] for row in processes)),
            wrong_gpu_allocations=wrong,passive_vulkan_other_gpus=passive,**bounds)
    finally:nv.nvmlShutdown()
