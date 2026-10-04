"""Parent-process RSS and CPU time of the parent plus its children.

Wall time is ``time.perf_counter``. CPU time has to include worker processes,
which disappear when the pool shuts down, so callers snapshot the process tree
while those children are still alive. Peak RSS is the parent only: that is the
process a user watches in Task Manager, and it is where unpickled partial
indexes land.
"""

from __future__ import annotations

import os
import sys
import threading


def gil_enabled() -> bool | None:
    """``sys._is_gil_enabled()`` on 3.13+, otherwise ``None``.

    Free-threaded builds (``3.13t``) return ``False``. A normal CPython returns
    ``True``. Interpreters older than 3.13 do not expose the probe.
    """
    probe = getattr(sys, "_is_gil_enabled", None)
    if probe is None:
        return None
    return bool(probe())


def current_rss() -> int:
    """Current resident set (working set) of this process, in bytes."""
    if sys.platform == "win32":
        return _win_rss(None)
    return _posix_rss()


def process_cpu_self() -> float:
    """User + kernel CPU seconds consumed by this process so far."""
    if sys.platform == "win32":
        return _win_cpu(None)
    return _posix_cpu_self()


def process_tree_cpu() -> float:
    """CPU seconds of this process plus every live descendant."""
    own = process_cpu_self()
    extra = 0.0
    if sys.platform == "win32":
        extra = _win_descendants_cpu(os.getpid())
    else:
        extra = _posix_descendants_cpu(os.getpid())
    return own + extra


class ResourceMonitor:
    """Sample the parent's RSS and the process tree's CPU during a build."""

    def __init__(self) -> None:
        self.peak_rss = 0
        self._self0 = 0.0
        self._tree0 = 0.0
        self._mid_tree: float | None = None
        self._mid_self: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._self0 = process_cpu_self()
        self._tree0 = process_tree_cpu()
        self.peak_rss = current_rss()
        self._thread = threading.Thread(
            target=self._sample, name="findex-rss", daemon=True
        )
        self._thread.start()

    def checkpoint(self) -> None:
        """CPU snapshot taken while worker processes are still alive.

        After ``ProcessPoolExecutor.shutdown`` the children are gone and their
        CPU time is no longer readable. Call this after ``future.result()``
        and before shutdown.
        """
        self._mid_tree = process_tree_cpu()
        self._mid_self = process_cpu_self()
        self._note_rss()

    def finish(self, *, children: bool) -> tuple[float, int]:
        """Return ``(cpu_seconds, peak_parent_rss)``."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        end_self = process_cpu_self()
        self._note_rss()
        with self._lock:
            peak = self.peak_rss
        if children and self._mid_tree is not None and self._mid_self is not None:
            # Parallel phase (parent + workers) plus the parent's merge afterwards.
            cpu = (self._mid_tree - self._tree0) + (end_self - self._mid_self)
        else:
            # Threads share this process, so its CPU clock already includes them.
            cpu = end_self - self._self0
        if cpu < 0:
            cpu = 0.0
        return cpu, peak

    def _sample(self) -> None:
        while not self._stop.wait(0.02):
            self._note_rss()

    def _note_rss(self) -> None:
        try:
            rss = current_rss()
        except OSError:
            return
        with self._lock:
            if rss > self.peak_rss:
                self.peak_rss = rss


def _posix_rss() -> int:
    try:
        parts = open("/proc/self/statm", encoding="ascii").read().split()
        return int(parts[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        import resource

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        if sys.platform == "darwin":
            return rss
        return rss * 1024


def _posix_cpu_self() -> float:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def _posix_descendants_cpu(root_pid: int) -> float:
    try:
        children = _linux_children_map()
    except OSError:
        return 0.0
    total = 0.0
    stack = list(children.get(root_pid, []))
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += _linux_pid_cpu(pid)
        stack.extend(children.get(pid, []))
    return total


def _linux_children_map() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    proc = "/proc"
    for name in os.listdir(proc):
        if not name.isdigit():
            continue
        try:
            stat = open(f"{proc}/{name}/stat", encoding="ascii").read()
        except OSError:
            continue
        # comm is wrapped in parentheses and may contain spaces.
        rparen = stat.rfind(")")
        fields = stat[rparen + 2 :].split()
        if len(fields) < 2:
            continue
        ppid = int(fields[1])  # state is fields[0], ppid is fields[1]
        children.setdefault(ppid, []).append(int(name))
    return children


def _linux_pid_cpu(pid: int) -> float:
    try:
        stat = open(f"/proc/{pid}/stat", encoding="ascii").read()
    except OSError:
        return 0.0
    rparen = stat.rfind(")")
    fields = stat[rparen + 2 :].split()
    # utime is field 12 (index 11) and stime field 13, in clock ticks,
    # counting from the first field after comm.
    if len(fields) < 13:
        return 0.0
    ticks = os.sysconf("SC_CLK_TCK")
    return (int(fields[11]) + int(fields[12])) / ticks


def _win_rss(pid: int | None) -> int:
    kernel32, psapi = _win_dlls()
    handle = _win_open(kernel32, pid)
    try:
        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = _ctypes.sizeof(counters)
        ok = psapi.GetProcessMemoryInfo(
            handle, _ctypes.byref(counters), counters.cb
        )
        if not ok:
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize)
    finally:
        _win_close(kernel32, handle, pid)


def _win_cpu(pid: int | None) -> float:
    kernel32, _psapi = _win_dlls()
    handle = _win_open(kernel32, pid)
    try:
        creation = _FILETIME()
        exit_time = _FILETIME()
        kernel = _FILETIME()
        user = _FILETIME()
        ok = kernel32.GetProcessTimes(
            handle,
            _ctypes.byref(creation),
            _ctypes.byref(exit_time),
            _ctypes.byref(kernel),
            _ctypes.byref(user),
        )
        if not ok:
            return 0.0
        return _filetime_seconds(kernel) + _filetime_seconds(user)
    finally:
        _win_close(kernel32, handle, pid)


def _win_descendants_cpu(root_pid: int) -> float:
    kernel32, _psapi = _win_dlls()
    children = _win_children(kernel32)
    total = 0.0
    stack = list(children.get(root_pid, []))
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            total += _win_cpu(pid)
        except OSError:
            continue
        stack.extend(children.get(pid, []))
    return total


def _win_children(kernel32: _WinDll) -> dict[int, list[int]]:
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid = _ctypes.cast(-1, _wintypes.HANDLE).value
    if not snapshot or snapshot == invalid:
        return {}
    children: dict[int, list[int]] = {}
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = _ctypes.sizeof(entry)
        ok = kernel32.Process32FirstW(snapshot, _ctypes.byref(entry))
        while ok:
            parent = int(entry.th32ParentProcessID)
            children.setdefault(parent, []).append(int(entry.th32ProcessID))
            ok = kernel32.Process32NextW(snapshot, _ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return children


def _win_open(kernel32: _WinDll, pid: int | None) -> int:
    if pid is None:
        return int(kernel32.GetCurrentProcess())
    # PROCESS_QUERY_LIMITED_INFORMATION is enough for GetProcessTimes.
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        raise OSError(f"OpenProcess failed for pid {pid}")
    return int(handle)


def _win_close(kernel32: _WinDll, handle: int, pid: int | None) -> None:
    # GetCurrentProcess() is a pseudo-handle and must not be closed.
    if pid is not None and handle:
        kernel32.CloseHandle(handle)


def _filetime_seconds(value: _FILETIME) -> float:
    ticks = (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)
    return ticks / 10_000_000


# ctypes is imported lazily so non-Windows tests don't pay for WinDLL setup
# at import, and so the names below exist for annotations on every platform.
import ctypes as _ctypes  # noqa: E402
from ctypes import wintypes as _wintypes  # noqa: E402

_WinDll = _ctypes.WinDLL


class _FILETIME(_ctypes.Structure):
    _fields_ = [
        ("dwLowDateTime", _wintypes.DWORD),
        ("dwHighDateTime", _wintypes.DWORD),
    ]


class _PROCESS_MEMORY_COUNTERS(_ctypes.Structure):
    _fields_ = [
        ("cb", _wintypes.DWORD),
        ("PageFaultCount", _wintypes.DWORD),
        ("PeakWorkingSetSize", _ctypes.c_size_t),
        ("WorkingSetSize", _ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", _ctypes.c_size_t),
        ("QuotaPagedPoolUsage", _ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", _ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", _ctypes.c_size_t),
        ("PagefileUsage", _ctypes.c_size_t),
        ("PeakPagefileUsage", _ctypes.c_size_t),
    ]


class _PROCESSENTRY32W(_ctypes.Structure):
    _fields_ = [
        ("dwSize", _wintypes.DWORD),
        ("cntUsage", _wintypes.DWORD),
        ("th32ProcessID", _wintypes.DWORD),
        ("th32DefaultHeapID", _ctypes.c_void_p),
        ("th32ModuleID", _wintypes.DWORD),
        ("cntThreads", _wintypes.DWORD),
        ("th32ParentProcessID", _wintypes.DWORD),
        ("pcPriClassBase", _ctypes.c_long),
        ("dwFlags", _wintypes.DWORD),
        ("szExeFile", _wintypes.WCHAR * 260),
    ]


_DLLS: tuple[_WinDll, _WinDll] | None = None


def _win_dlls() -> tuple[_WinDll, _WinDll]:
    global _DLLS
    if _DLLS is not None:
        return _DLLS
    kernel32 = _ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = _ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = _wintypes.HANDLE
    kernel32.OpenProcess.restype = _wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [_wintypes.DWORD, _wintypes.BOOL, _wintypes.DWORD]
    kernel32.CloseHandle.argtypes = [_wintypes.HANDLE]
    kernel32.GetProcessTimes.argtypes = [
        _wintypes.HANDLE,
        _ctypes.POINTER(_FILETIME),
        _ctypes.POINTER(_FILETIME),
        _ctypes.POINTER(_FILETIME),
        _ctypes.POINTER(_FILETIME),
    ]
    kernel32.GetProcessTimes.restype = _wintypes.BOOL
    kernel32.CreateToolhelp32Snapshot.restype = _wintypes.HANDLE
    kernel32.CreateToolhelp32Snapshot.argtypes = [_wintypes.DWORD, _wintypes.DWORD]
    kernel32.Process32FirstW.argtypes = [
        _wintypes.HANDLE,
        _ctypes.POINTER(_PROCESSENTRY32W),
    ]
    kernel32.Process32FirstW.restype = _wintypes.BOOL
    kernel32.Process32NextW.argtypes = [
        _wintypes.HANDLE,
        _ctypes.POINTER(_PROCESSENTRY32W),
    ]
    kernel32.Process32NextW.restype = _wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        _wintypes.HANDLE,
        _ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
        _wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = _wintypes.BOOL
    _DLLS = (kernel32, psapi)
    return _DLLS
