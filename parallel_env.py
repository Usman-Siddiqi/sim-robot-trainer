"""
parallel_env.py

Author: Usman Siddiqi

Runs several copies ("shards") of an environment side by side, each in its own
process, and presents them as one big environment.

Why: one environment steps its physics on every core, but then computes the
observations and rewards of all robots in a single Python thread - and that
part takes about as long as the physics, so most cores sit idle half the time.
Python threads do not help (only one runs Python at a time), separate processes
do: every shard has its own interpreter and its own share of the physics
threads.

    env = ParallelEnv("hike", num_envs=4096, shards=4, num_threads=16)

behaves like one HikeEnv with 4096 robots.
"""

import multiprocessing as mp
import os

import numpy as np


def make_single(task, **kwargs):
    """One plain environment of a task ("hike" or "sort")."""
    if task == "hike":
        from env_hike import HikeEnv
        return HikeEnv(**kwargs)
    if task == "sort":
        from config import SortCfg
        from env_sort import SortEnv
        kwargs.pop("flat_only", None)
        return SortEnv(guide=SortCfg.guide_start, mass_level=SortCfg.mass_start, **kwargs)
    raise ValueError(f"unknown task '{task}' (hike or sort)")


def _worker(conn, task, kwargs):
    """Body of a shard's process: build the environment, then do what the main process asks for."""
    try:
        env = make_single(task, **kwargs)
        conn.send({name: getattr(env, name) for name in
                   ("num_envs", "num_threads", "num_obs", "num_actions", "task", "dt", "obs_layout",
                    "progress_keys", "max_episode_length")})
        while True:
            command, args = conn.recv()
            if command == "close":
                env.close()
                break
            conn.send(getattr(env, command)(*args))
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        pass                                                # the main process is gone: just stop


class ParallelEnv:
    def __init__(self, task, num_envs, shards, num_threads=None, seed=0, **kwargs):
        """
        Args:
            task: "hike" or "sort"
            num_envs: robots in total (split evenly; the remainder is dropped)
            shards: number of processes
            num_threads: physics threads in total (default: all logical cores)
            kwargs: passed on to every environment
        """
        threads = int(num_threads or os.cpu_count() or 1)
        shards = max(1, min(int(shards), threads))
        per = int(num_envs) // shards
        context = mp.get_context("spawn")
        self._conns, self._procs = [], []
        for k in range(shards):
            parent, child = context.Pipe()
            # every shard gets its own random numbers, but all build the same world
            options = dict(kwargs, num_envs=per, num_threads=max(1, threads // shards), seed=seed + 7919 * k)
            proc = context.Process(target=_worker, args=(child, task, options), daemon=True)
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)
        facts = [conn.recv() for conn in self._conns]       # waits until every shard is built
        self._slices = [slice(k * per, (k + 1) * per) for k in range(shards)]
        self.num_envs = per * shards
        self.num_threads = sum(f["num_threads"] for f in facts)
        for name in ("num_obs", "num_actions", "task", "dt", "obs_layout", "progress_keys", "max_episode_length"):
            setattr(self, name, facts[0][name])

    def _ask(self, command, *args, per_shard=None):
        """Send a command to every shard, then collect the answers (the shards work at the same time)."""
        for k, conn in enumerate(self._conns):
            conn.send((command, per_shard[k] if per_shard is not None else args))
        return [conn.recv() for conn in self._conns]

    def reset(self):
        return np.concatenate(self._ask("reset"))

    def step(self, actions):
        results = self._ask("step", per_shard=[(actions[s],) for s in self._slices])
        obs = np.concatenate([r[0] for r in results])
        reward = np.concatenate([r[1] for r in results])
        done = np.concatenate([r[2] for r in results])
        info = {}
        for key in results[0][3]:
            if key != "episode":
                info[key] = np.concatenate([r[3][key] for r in results])
        # statistics of the episodes that just ended: averaged over the shards that had any
        stats = [r[3]["episode"] for r in results if "episode" in r[3]]
        if stats:
            keys = set().union(*stats)
            info["episode"] = {k: float(np.mean([s[k] for s in stats if k in s])) for k in keys}
        return obs, reward, done, info

    # ------------------------------------------------- things the training loop asks for
    def curriculum_state(self):
        states = self._ask("curriculum_state")
        merged = dict(states[0])
        for key, value in merged.items():
            if isinstance(value, list):                     # per-robot lists (terrain levels): all shards
                merged[key] = [x for s in states for x in s[key]]
        return merged

    def load_curriculum(self, state, rng):
        self._ask("load_curriculum", state, np.random.default_rng(int(rng.integers(1 << 30))))

    def on_iteration(self, iteration, stats):
        self._ask("on_iteration", iteration, stats)

    def close(self):
        for conn in self._conns:
            try:
                conn.send(("close", ()))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=10)
