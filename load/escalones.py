"""Forma de carga por escalones para la prueba de estrés.

Se combina con el locustfile: locust -f load/locustfile.py,load/escalones.py
Cada escalón sostiene la carga el tiempo suficiente para medirla estable.
"""

import os

from locust import LoadTestShape


class Escalones(LoadTestShape):
    step_users = 20
    step_seconds = 30
    max_users = int(os.environ.get("LOAD_MAX_USERS", "200"))
    spawn_rate = 10

    def tick(self):
        step = int(self.get_run_time() // self.step_seconds) + 1
        users = step * self.step_users
        if users > self.max_users:
            return None
        return users, self.spawn_rate
